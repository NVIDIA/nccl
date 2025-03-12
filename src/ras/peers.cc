/*************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#define NDEBUG // Comment out during development only!
#include <cassert>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "nccl.h"
#include "ras_internal.h"


// All the known peer NCCL processes. The array is sorted by addr to ensure locality (within a node and hopefully
// also within a DC).  The array may grow over time and it *includes* dead peers.
struct rasPeerInfo* rasPeers;
int nRasPeers;
// Hash of the rasPeers array, for figuring out when to sync with a remote peer.
uint64_t rasPeersHash;
// Index of this process within the rasPeers array (may change over time as the array grows).
static int myPeerIdx = -1;

// Addresses of all the dead peers, sorted.  In principle we could instead have a flag in rasPeerInfo for this,
// but we expect rasPeers to be largely static (and large at scale!) and rasDeadPeers to be fairly dynamic and
// much smaller, so we prefer to keep the dead info separately so that we don't end up sending the possibly large
// rasPeerInfo array around all the time.
union ncclSocketAddress* rasDeadPeers;
// The number of dead peers.
int nRasDeadPeers;
// The array size (may be larger than nRasDeadPeers).
static int rasDeadPeersSize;
// Hash of the rasDeadPeers array, for figuring out when to sync with a remote peer.
uint64_t rasDeadPeersHash;

static ncclResult_t rasRanksConvertToPeers(struct rasRankInit* ranks, int nranks,
                                           struct rasPeerInfo** rankPeers, int *nRankPeers, int* newNRasPeers);
static ncclResult_t rasPeersUpdate(struct rasPeerInfo* rankPeers, int* nRankPeers, int newNRasPeers = -1);

static ncclResult_t rasNetUpdatePeers(const struct rasPeerInfo* newPeers, int nNewPeers, bool updateDeadPeers,
                                      struct rasRankInit* ranks = nullptr, int nranks = 0,
                                      struct rasConnection* fromConn = nullptr);
static ncclResult_t rasLinkPropagateUpdate(struct rasLink* link, const struct rasPeerInfo* newPeers, int nNewPeers,
                                           bool updateDeadPeers, struct rasRankInit* ranks, int nranks,
                                           struct rasConnection* fromConn);
static ncclResult_t rasConnPropagateUpdate(struct rasConnection* conn, const struct rasPeerInfo* newPeers,
                                           int nNewPeers, bool updateDeadPeers, struct rasRankInit* ranks, int nranks);
ncclResult_t rasMsgHandlePeersUpdate(struct rasMsg* msg, struct rasSocket* sock);

static ncclResult_t rasLinkReinitConns(struct rasLink* link);

static ncclResult_t rasDeadPeersUpdate(union ncclSocketAddress* updatePeers, int* nUpdatePeers);
static ncclResult_t getNewDeadEntry(union ncclSocketAddress** pAddr);

static int rasAddrRankInitCompare(const void* k, const void* e);
static int rasAddrPeerInfoCompare(const void* k, const void* e);
static int rasRanksCompare(const void* e1, const void* e2);

static void rasPeersDump();
static void rasDeadPeersDump();
static char* rasPeerDump(const struct rasPeerInfo* peer, char* result, size_t nres);


/////////////////////////////////////////////////////////////////////////////
// Functions related to the handling of local RAS_ADD_RANKS notifications. //
/////////////////////////////////////////////////////////////////////////////

// Handles RAS_ADD_RANKS notification -- adds new ranks to the internal list of all RAS peers, reconfigures RAS
// network connections, and notifies the peers.
ncclResult_t rasLocalHandleAddRanks(struct rasRankInit* ranks, int nranks) {
  ncclResult_t ret = ncclSuccess;

  INFO(NCCL_RAS, "RAS handling local addRanks request (old nRasPeers %d)", nRasPeers);

  // Convert the input rasRankInit structures into our internal rasPeerInfo.
  struct rasPeerInfo* rankPeers = nullptr;
  int nRankPeers;
  int newNRasPeers;
  NCCLCHECKGOTO(rasRanksConvertToPeers(ranks, nranks, &rankPeers, &nRankPeers, &newNRasPeers), ret, fail);

  // Update local rasPeers.
  NCCLCHECKGOTO(rasPeersUpdate(rankPeers, &nRankPeers, newNRasPeers), ret, fail);

  INFO(NCCL_RAS, "RAS finished local processing of addRanks request (new nRasPeers %d, nRankPeers %d)",
       nRasPeers, nRankPeers);
  // Print peers only if something changed and we're the "root".
  if (nRankPeers > 0 && memcmp(&ranks[0].addr, &rasNetListeningSocket.addr, sizeof(ranks[0].addr)) == 0)
    rasPeersDump();

  // Propagate the changes through our RAS network links.
  NCCLCHECKGOTO(rasNetUpdatePeers(rankPeers, nRankPeers, /*updateDeadPeers*/false, ranks, nranks), ret, fail);

exit:
  if (rankPeers)
    free(rankPeers);
  free(ranks);
  return ret;
fail:
  goto exit;
}

// Converts the rasRankInit structure into rasPeerInfo.  This skips empty elements (in case of errors), orders
// elements by the address/cudaDev, and merges elements with duplicate addresses (in case of multiple CUDA devices per
// process).  In the process we also calculate how large the merged rasPeers array will need to be.
static ncclResult_t rasRanksConvertToPeers(struct rasRankInit* ranks, int nranks,
                                           struct rasPeerInfo** rankPeers, int *nRankPeers, int* newNRasPeers) {
  ncclResult_t ret = ncclSuccess;
  int peerIdx, rankPeerIdx;

  // Handy when checking for empty (in case of errors) addresses.
  union ncclSocketAddress emptyAddr;
  memset(&emptyAddr, '\0', sizeof(emptyAddr));

  // Begin by sorting the array by address and cudaDev (to match the rasPeers order).
  qsort(ranks, nranks, sizeof(*ranks), &rasRanksCompare);

  // We over-allocate peers here because to get an accurate count we would need to loop over the ranks first...
  // nRankPeers will hold the actual count of used elements.
  *rankPeers = nullptr;
  NCCLCHECKGOTO(ncclCalloc(rankPeers, nranks), ret, fail);

  peerIdx = rankPeerIdx = 0;
  *newNRasPeers = nRasPeers;
  for (int rankIdx = 0; rankIdx < nranks; rankIdx++) {
    const struct rasRankInit* rank = ranks+rankIdx;
    struct rasPeerInfo* rankPeer = *rankPeers+rankPeerIdx;

    if (memcmp(&emptyAddr, &rank->addr, sizeof(emptyAddr)) == 0) {
      // Skip empty rank entries.
      continue;
    }

    // First check if the rank doesn't need to be merged into the previous entry in rankPeers
    // (possible if there are multiple ranks with the same address).
    if (rankPeerIdx > 0 && memcmp(&rank->addr, &rankPeer[-1].addr, sizeof(rank->addr)) == 0) {
      // Merge into the previous entry in peers.
      rankPeer[-1].cudaDevs |= (1UL << rank->cudaDev);
      rankPeer[-1].nvmlDevs |= (1UL << rank->nvmlDev);
      continue;
    }

    // Add a new entry to rankPeers.
    assert(rankPeerIdx < nranks);
    memcpy(&rankPeer->addr, &rank->addr, sizeof(rankPeer->addr));
    rankPeer->pid = rank->pid;
    rankPeer->cudaDevs = (1UL << rank->cudaDev);
    rankPeer->nvmlDevs = (1UL << rank->nvmlDev);
    rankPeer->hostHash = rank->hostHash;
    rankPeer->pidHash = rank->pidHash;
    rankPeerIdx++;

    // Also check if there is already an entry with that address in the global rasPeers so that the caller can know how
    // many more entries will be needed.
    const struct rasPeerInfo* rasPeer = rasPeers+peerIdx;
    int cmp = 0;
    while (peerIdx < nRasPeers) {
      cmp = ncclSocketsCompare(&rank->addr, &rasPeer->addr);
      if (cmp <= 0)
        break;
      peerIdx++;
      rasPeer++;
    }
    if (peerIdx == nRasPeers) {
      // The current rank is "greater than" all existing peers, so it will need a new entry.  We stay in the loop so
      // that we don't need to handle the remaining ranks separately.
      (*newNRasPeers)++;
      continue;
    }
    if (cmp < 0) {
      (*newNRasPeers)++;
    } else {
      // Duplicates (cmp == 0) between the rank array and the peers array will be merged.
      assert(rank->pid == rasPeer->pid);
    }
  }
  assert(peerIdx <= nRasPeers);
  *nRankPeers = rankPeerIdx;

exit:
  return ret;
fail:
  if (*rankPeers) {
    free(*rankPeers);
    *rankPeers = nullptr;
  }
  goto exit;
}

// Updates the rasPeers array with the new data.  The new data gets updated in the process as well: any data that
// wasn't actually new is purged, so as to minimize the amount of data we forward to our peers.
// On a successful return, nRankPeers contains the number of entries that were updated.
static ncclResult_t rasPeersUpdate(struct rasPeerInfo* rankPeers, int* nRankPeers, int newNRasPeers) {
  ncclResult_t ret = ncclSuccess;
  int rankPeerIdxDst;
  int rankPeerIdx, peerIdx;

  if (newNRasPeers == -1) {
    // First calculate the new size of rasPeers.
    newNRasPeers = nRasPeers;
    for (rankPeerIdx = peerIdx = 0; rankPeerIdx < *nRankPeers; rankPeerIdx++) {
      struct rasPeerInfo* rankPeer = rankPeers+rankPeerIdx;
      struct rasPeerInfo* rasPeer = rasPeers+peerIdx;
      int cmp = 1;

      while (peerIdx < nRasPeers) {
        cmp = ncclSocketsCompare(&rankPeer->addr, &rasPeer->addr);

        if (cmp < 0) {
          // rankPeer will go in front of rasPeer.
          newNRasPeers++;
          break;
        }

        peerIdx++;
        rasPeer++;

        if (cmp == 0)
          break;
      }
      if (cmp > 0) // No more rasPeer entries -- rankPeer will go at the end.
        newNRasPeers++;
    }
  }

  // If needed, allocate a new, larger rasPeers array.
  struct rasPeerInfo* newRasPeers;
  int myNewPeerIdx;
  if (newNRasPeers > nRasPeers) {
    NCCLCHECKGOTO(ncclCalloc(&newRasPeers, newNRasPeers), ret, fail);
  } else {
    newRasPeers = rasPeers;
  }

  // Now merge the rankPeers into newRasPeers.  In the process, modify rankPeers to become a "diff" between
  // the old rasPeers and newRasPeers -- this will be the data structure to broadcast on the RAS network.
  myNewPeerIdx = -1;
  int newPeerIdx;
  for (newPeerIdx = rankPeerIdx = peerIdx = 0; rankPeerIdx < *nRankPeers || peerIdx < nRasPeers;) {
    struct rasPeerInfo* rankPeer = rankPeers+rankPeerIdx;
    struct rasPeerInfo* rasPeer = rasPeers+peerIdx;
    struct rasPeerInfo* newRasPeer = newRasPeers+newPeerIdx;

    if (rankPeerIdx < *nRankPeers) {
      if (peerIdx < nRasPeers) {
        int cmp = ncclSocketsCompare(&rankPeer->addr, &rasPeer->addr);

        if (cmp < 0) {
          // rankPeer needs to occur before rasPeer -- that's possible only if we are adding new entries.
          assert(newRasPeers != rasPeers);
          // Add new entry to newRasPeers.
          assert(newPeerIdx < newNRasPeers);
          memcpy(newRasPeer, rankPeer, sizeof(*newRasPeer));
          newPeerIdx++;
          rankPeerIdx++;
        }
        else {
          // cmp >= 0 -- Start by copying peer to newRasPeer, if needed.
          if (newRasPeers != rasPeers) {
            assert(newPeerIdx < newNRasPeers);
            memcpy(newRasPeer, rasPeer, sizeof(*newRasPeer));
          }
          else { // in-place
            assert(newRasPeer == rasPeer);
          }

          if (cmp == 0) {
            // The address of rankPeer is the same as that of newRasPeer -- merge into it.
            // First though calculate what GPUs from rankPeer are actually new (if any).
            uint64_t newDevs = rankPeer->cudaDevs & ~newRasPeer->cudaDevs;
            newRasPeer->cudaDevs |= rankPeer->cudaDevs;
            // Update rankPeer->devs with the newly added devs only -- we'll clean it up at the end.
            rankPeer->cudaDevs = newDevs;
            // Repeat for nvmlDevs...
            newDevs = rankPeer->nvmlDevs & ~newRasPeer->nvmlDevs;
            newRasPeer->nvmlDevs |= rankPeer->nvmlDevs;
            rankPeer->nvmlDevs = newDevs;
            rankPeerIdx++;
          }
          // Given that we might've added new entries, we need to update myPeerIdx as well.
          if (myPeerIdx == peerIdx)
            myNewPeerIdx = newPeerIdx;
          peerIdx++;
          newPeerIdx++;
        }
      } else { // peerIdx == nRasPeers
        // No more rasPeers -- add a new entry based on rank.
        assert(newPeerIdx < newNRasPeers);
        memcpy(newRasPeer, rankPeer, sizeof(*newRasPeer));
        // If this is the first time this function is run, myPeerIdx will need to be set.  It's more work in that
        // case as we need to compare the addresses of each peer until we find one.
        if (myPeerIdx == -1 && memcmp(&newRasPeer->addr, &rasNetListeningSocket.addr, sizeof(newRasPeer->addr)) == 0)
          myNewPeerIdx = newPeerIdx;
        newPeerIdx++;
        rankPeerIdx++;
      }
    } else { // rankPeerIdx == *nRankPeers
      // No more rankPeers -- copy the rasPeer over if needed.
      if (newRasPeers != rasPeers) {
        assert(newPeerIdx < newNRasPeers);
        memcpy(newRasPeer, rasPeer, sizeof(*newRasPeer));
      }
      else { // in-place at the end.
        assert(newRasPeer == rasPeer);
      }
      if (myPeerIdx == peerIdx)
        myNewPeerIdx = newPeerIdx;
      peerIdx++;
      newPeerIdx++;
    }
  }
  assert(newPeerIdx == newNRasPeers);

  if (newRasPeers != rasPeers) {
    if (rasPeers)
      free(rasPeers);
    rasPeers = newRasPeers;
    nRasPeers = newNRasPeers;
    assert(myNewPeerIdx != -1);
    myPeerIdx = myNewPeerIdx;
  } else {
    assert(myNewPeerIdx == myPeerIdx);
  }
  rasPeersHash = getHash((const char*)rasPeers, nRasPeers*sizeof(*rasPeers));

  // Purge from rankPeers all entries that didn't actually contribute any new GPUs.
  for (rankPeerIdx = rankPeerIdxDst = 0; rankPeerIdx < *nRankPeers; rankPeerIdx++) {
    struct rasPeerInfo* rankPeer = rankPeers+rankPeerIdx;
    if (rankPeer->cudaDevs != 0) {
      if (rankPeerIdxDst != rankPeerIdx) {
        memcpy(rankPeers+rankPeerIdxDst, rankPeer, sizeof(*rankPeers));
      }
      rankPeerIdxDst++;
    }
  }
  assert(rankPeerIdxDst <= *nRankPeers);
  *nRankPeers = rankPeerIdxDst;

exit:
  return ret;
fail:
  goto exit;
}

// Searches through rasPeers given the peer address.  Returns the index of the found entry in the rasPeers
// array or -1 if not found.
int rasPeerFind(const union ncclSocketAddress* addr) {
  struct rasPeerInfo* peer = (struct rasPeerInfo*)bsearch(addr, rasPeers, nRasPeers, sizeof(*rasPeers),
                                                          rasAddrPeerInfoCompare);
  return (peer ? peer-rasPeers : -1);
}


/////////////////////////////////////////////////////////////////////////////////
// Functions related to the propagation of peers updates over the RAS network. //
/////////////////////////////////////////////////////////////////////////////////

// Propagates information about new peers through the RAS network links.
// ranks -- if provided -- lists all the peers who are already aware of this update (because they are the members
// of the new communicator being established), and who thus don't need to be notified.  updatedDeadPeers can
// be used, however, to request at least the propagation of rasDeadPeers to such peers.
// fromConn -- if provided -- identifies the connection used to receive this update; there's no need to
// propagate the update back through it.
// Reconfigures the RAS network to accommodate the newly added peers, by modifying the links and establishing new
// connections as needed.
static ncclResult_t rasNetUpdatePeers(const struct rasPeerInfo* newPeers, int nNewPeers, bool updateDeadPeers,
                                      struct rasRankInit* ranks, int nranks, struct rasConnection* fromConn) {
  ncclResult_t ret = ncclSuccess;

  // Do we actually have anything to do?
  if (nNewPeers == 0 && !updateDeadPeers)
    goto exit;

  // Start by propagating the update through the RAS network links.  We consider any errors during this process
  // to be non-fatal (we can re-sync later around a keep-alive exchange).
  (void)rasLinkPropagateUpdate(&rasNextLink, newPeers, nNewPeers, updateDeadPeers, ranks, nranks, fromConn);
  (void)rasLinkPropagateUpdate(&rasPrevLink, newPeers, nNewPeers, updateDeadPeers, ranks, nranks, fromConn);

  // Calculate new link peers and open new connections if needed.
  NCCLCHECKGOTO(rasLinkReinitConns(&rasNextLink), ret, fail);
  NCCLCHECKGOTO(rasLinkReinitConns(&rasPrevLink), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

// Sends a peers update through all the connections associated with a particular link.  See rasNetUpdatePeers
// for the explanation of the function arguments.
static ncclResult_t rasLinkPropagateUpdate(struct rasLink* link, const struct rasPeerInfo* newPeers, int nNewPeers,
                                           bool updateDeadPeers, struct rasRankInit* ranks, int nranks,
                                           struct rasConnection* fromConn) {
  for (struct rasLinkConn* linkConn = link->conns; linkConn; linkConn = linkConn->next) {
    // Note that we don't send the update via the connection that we received this notification from in the first
    // place (while it wouldn't loop indefinitely, it would add a needless extra exchange).
    if (linkConn->conn && linkConn->conn != fromConn) {
      // Failed propagations are not considered fatal (we will retry after a keep-alive).
      (void)rasConnPropagateUpdate(linkConn->conn, newPeers, nNewPeers, updateDeadPeers, ranks, nranks);
    }
  }

  return ncclSuccess;
}

// Sends a peers update down a particular connection.  See rasNetUpdatePeers for the explanation of the function
// arguments.
static ncclResult_t rasConnPropagateUpdate(struct rasConnection* conn, const struct rasPeerInfo* newPeers,
                                           int nNewPeers, bool updateDeadPeers, struct rasRankInit* ranks, int nranks) {
  if (conn->sock && conn->sock->status == RAS_SOCK_READY) {
    // If we have the rank info, check if the peer on the other side of this connection has participated in the new
    // communicator.
    int connRank = -1;
    if (ranks && !updateDeadPeers) {
      struct rasRankInit* rank = (struct rasRankInit*)bsearch(&conn->addr, ranks, nranks, sizeof(*ranks),
                                                              rasAddrRankInitCompare);
      if (rank)
        connRank = rank-ranks;
    }
    if (connRank < 0) {
      // It did not participate or we don't know -- we should send an update to that peer then.
      NCCLCHECK(rasConnSendPeersUpdate(conn, newPeers, nNewPeers));
    }
  }

  return ncclSuccess;
}

// Sends a RAS_MSG_PEERSUPDATE message, which can include both the rasPeers (preferably only the newly added peers
// rather than the complete rasPeers array, to save on the network bandwidth) and rasDeadPeers (sent in its entirety
// if at all, as it's assumed to be a lot smaller than rasPeers).
ncclResult_t rasConnSendPeersUpdate(struct rasConnection* conn, const struct rasPeerInfo* peers, int nPeers) {
  struct rasMsg* msg = nullptr;
  int msgLen;
  int deadPeersOffset = 0;
  int nDeadPeers;

  if (conn->lastSentPeersHash == rasPeersHash || conn->lastRecvPeersHash == rasPeersHash) {
    nPeers = 0;
  }
  if (conn->lastSentDeadPeersHash == rasDeadPeersHash || conn->lastRecvDeadPeersHash == rasDeadPeersHash) {
    nDeadPeers = 0;
  } else {
    // We expect the rasDeadPeers array to be much smaller than rasPeers so if we send it, we send it in full.
    nDeadPeers = nRasDeadPeers;
  }

  if (nPeers == 0 && nDeadPeers == 0)
    goto exit;

  msgLen = rasMsgLength(RAS_MSG_PEERSUPDATE) + nPeers*sizeof(*peers);
  if (nDeadPeers > 0) {
    ALIGN_SIZE(msgLen, alignof(union ncclSocketAddress));
    deadPeersOffset = msgLen;
    msgLen += nDeadPeers*sizeof(*rasDeadPeers);
  }

  NCCLCHECK(rasMsgAlloc(&msg, msgLen));
  msg->type = RAS_MSG_PEERSUPDATE;
  msg->peersUpdate.peersHash = rasPeersHash;
  msg->peersUpdate.nPeers = nPeers;
  msg->peersUpdate.deadPeersHash = rasDeadPeersHash;
  msg->peersUpdate.nDeadPeers = nDeadPeers;
  memcpy(msg->peersUpdate.peers, peers, nPeers * sizeof(msg->peersUpdate.peers[0]));
  if (nDeadPeers > 0)
    memcpy(((char*)msg)+deadPeersOffset, rasDeadPeers, nDeadPeers * sizeof(*rasDeadPeers));

  if (nPeers > 0)
    conn->lastSentPeersHash = rasPeersHash;
  if (nDeadPeers > 0)
    conn->lastSentDeadPeersHash = rasDeadPeersHash;

  INFO(NCCL_RAS, "RAS sending a peersUpdate to %s (nPeers %d, nDeadPeers %d)",
       ncclSocketToString(&conn->addr, rasLine), nPeers, nDeadPeers);

  rasConnEnqueueMsg(conn, msg, msgLen);
exit:
  return ncclSuccess;
}

// Handles the RAS_MSG_PEERSUPDATE message on the receiver side.  The received data is merged into the local
// rasPeers and rasDeadPeers arrays.  If the checksums of the resulting arrays don't match those from the message,
// sends its own RAS_MSG_PEERSUPDATE back to the source, to ensure a sync.
// Subsequently propagates the update to its own peers.
ncclResult_t rasMsgHandlePeersUpdate(struct rasMsg* msg, struct rasSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct rasMsg* newMsg = nullptr;
  int newMsgLen = 0;
  assert(sock->conn);
  int nPeers, nDeadPeers;
  int deadPeersOffset = 0;
  bool updatePeers, updateDeadPeers;

  INFO(NCCL_RAS, "RAS handling peersUpdate from %s (peersHash 0x%lx, deadPeersHash 0x%lx, nPeers %d, nDeadPeers %d)",
       ncclSocketToString(&sock->sock.addr, rasLine), msg->peersUpdate.peersHash, msg->peersUpdate.deadPeersHash,
       msg->peersUpdate.nPeers, msg->peersUpdate.nDeadPeers);
  INFO(NCCL_RAS, "RAS my old rasPeersHash 0x%lx, rasDeadPeersHash 0x%lx, nRasPeers %d, nRasDeadPeers %d",
       rasPeersHash, rasDeadPeersHash, nRasPeers, nRasDeadPeers);
  sock->conn->lastRecvPeersHash = msg->peersUpdate.peersHash;
  sock->conn->lastRecvDeadPeersHash = msg->peersUpdate.deadPeersHash;

  // Prepare ours to send back.  We don't enqueue it right away because we want to make sure first that we need
  // to send it.  We'll find out by comparing the hash values after the merge.
  // We want to prepare the message pre-merge though because post-merge it will include the just received new peers,
  // and it's pointless to send those back to where they just came from.
  // nPeers and nDeadPeers are used primarily for message length calculations, so they have to assume the worst-case
  // scenario (e.g., no overlap in case of nDeadPeers).
  nPeers = (msg->peersUpdate.peersHash != rasPeersHash ? nRasPeers : 0);
  nDeadPeers = (msg->peersUpdate.deadPeersHash != rasDeadPeersHash ? nRasDeadPeers+msg->peersUpdate.nDeadPeers : 0);
  if (nPeers > 0 || nDeadPeers > 0) {
    newMsgLen = rasMsgLength(RAS_MSG_PEERSUPDATE) + nPeers*sizeof(*rasPeers);
    if (nDeadPeers > 0) {
      ALIGN_SIZE(newMsgLen, alignof(union ncclSocketAddress));
      newMsgLen += nDeadPeers*sizeof(*rasDeadPeers);
    }
    NCCLCHECKGOTO(rasMsgAlloc(&newMsg, newMsgLen), ret, fail);
    newMsg->type = RAS_MSG_PEERSUPDATE;
    // Note that after rasPeersUpdate below we may still decide not to send the peers.
    memcpy(newMsg->peersUpdate.peers, rasPeers, nPeers * sizeof(newMsg->peersUpdate.peers[0]));
    newMsg->peersUpdate.nPeers = nPeers;

    if (nDeadPeers > 0) {
      // Calculate the offset where dead peers are stored in the received message.  We do it before the peers
      // update because it could modify msg->peersUpdate.nPeers...
      deadPeersOffset = rasMsgLength(RAS_MSG_PEERSUPDATE) + msg->peersUpdate.nPeers * sizeof(msg->peersUpdate.peers[0]);
      ALIGN_SIZE(deadPeersOffset, alignof(union ncclSocketAddress));
    }

    if (nPeers > 0)
      NCCLCHECKGOTO(rasPeersUpdate(msg->peersUpdate.peers, &msg->peersUpdate.nPeers), ret, fail);
    else
      msg->peersUpdate.nPeers = 0;
    if (nDeadPeers > 0)
      NCCLCHECKGOTO(rasDeadPeersUpdate((union ncclSocketAddress*)(((char*)msg)+deadPeersOffset),
                                       &msg->peersUpdate.nDeadPeers), ret, fail);
    else
      msg->peersUpdate.nDeadPeers = 0;

    INFO(NCCL_RAS, "RAS finished local processing of peersUpdate "
         "(new nRasPeers %d, nRasDeadPeers %d, nPeers %d, nDeadPeers %d)",
         nRasPeers, nRasDeadPeers, msg->peersUpdate.nPeers, msg->peersUpdate.nDeadPeers);
    if (msg->peersUpdate.nPeers > 0)
      rasPeersDump();
    if (msg->peersUpdate.nDeadPeers > 0)
      rasDeadPeersDump();

    // If post-merge the hashes are still different, send our (dead) peers back.
    updatePeers = (sock->conn->lastSentPeersHash != rasPeersHash && sock->conn->lastRecvPeersHash != rasPeersHash);
    updateDeadPeers = (sock->conn->lastSentDeadPeersHash != rasDeadPeersHash &&
                       sock->conn->lastRecvDeadPeersHash != rasDeadPeersHash);
    if (updatePeers || updateDeadPeers) {
      newMsg->peersUpdate.peersHash = rasPeersHash;
      newMsg->peersUpdate.deadPeersHash = rasDeadPeersHash;
      if (updatePeers) {
        assert(nPeers > 0);
        sock->conn->lastSentPeersHash = rasPeersHash;
      } else {
        // If hashes match, make sure that we don't send the rasPeers back.
        newMsg->peersUpdate.nPeers = 0;
      }

      // We need to recalculate the message size from scratch now that both rasPeers and rasDeadPeers may have changed.
      newMsgLen = rasMsgLength(RAS_MSG_PEERSUPDATE) + newMsg->peersUpdate.nPeers * sizeof(*rasPeers);

      if (updateDeadPeers) {
        assert(nRasDeadPeers > 0);
        sock->conn->lastSentDeadPeersHash = rasDeadPeersHash;

        ALIGN_SIZE(newMsgLen, alignof(union ncclSocketAddress));
        deadPeersOffset = newMsgLen;
        newMsgLen += nRasDeadPeers*sizeof(*rasDeadPeers);

        memcpy(((char*)newMsg)+deadPeersOffset, rasDeadPeers, nDeadPeers * sizeof(*rasDeadPeers));
        sock->conn->lastSentDeadPeersHash = rasDeadPeersHash;
        newMsg->peersUpdate.nDeadPeers = nRasDeadPeers;
      } else {
        newMsg->peersUpdate.nDeadPeers = 0;
      }

      INFO(NCCL_RAS, "RAS sending back a peersUpdate (nPeers %d, nDeadPeers %d)",
           newMsg->peersUpdate.nPeers, newMsg->peersUpdate.nDeadPeers);

      rasConnEnqueueMsg(sock->conn, newMsg, newMsgLen);
      newMsg = nullptr;
    } // if (updatePeers || updateDeadPeers)

    // Propagate the changes through our RAS network links.
    NCCLCHECKGOTO(rasNetUpdatePeers(msg->peersUpdate.peers, msg->peersUpdate.nPeers, updateDeadPeers, nullptr, 0,
                                    sock->conn), ret, fail);
  }

exit:
  rasMsgFree(newMsg);
  return ret;
fail:
  goto exit;
}


//////////////////////////////////////////////////////////////////////////////////////////
// Functions related to the (re-)configuration of RAS connections after a peers update. //
//////////////////////////////////////////////////////////////////////////////////////////

// Reinitializes the connection(s) of a particular link, following a peers update.
// Adding new peers can affect the calculation of the link's primary connection and also the fallbacks.
// The newly added peers could also shift all the existing peerIdx values, invalidating the values in rasLinkConn
// structures, so it's better to drop it all and recalculate from scratch.
// We recalculate the primary peer; if an active connection to it already exists, then we're done.  If there
// is no connection, we create one.  If a connection exists but is experiencing delays then we add a fallback and
// the process repeats.
// External conns are dropped from the links as well (they will be re-created via keepAlive messages as needed).
static ncclResult_t rasLinkReinitConns(struct rasLink* link) {
  struct rasLinkConn* linkConn;
  int newPeerIdx = myPeerIdx;

  if (link->conns) {
    // Free the old contents but keep the first entry for convenience (though wipe it).
    for (struct rasLinkConn* linkConn = link->conns->next; linkConn;) {
      struct rasLinkConn* linkConnNext = linkConn->next;
      free(linkConn);
      linkConn = linkConnNext;
    }
    memset(link->conns, '\0', sizeof(*link->conns));
    link->lastUpdatePeersTime = 0;
  } else { // link->conns == nullptr
    NCCLCHECK(ncclCalloc(&link->conns, 1));
  }

  // Fill in the entry for the primary connection.
  linkConn = link->conns;
  linkConn->peerIdx = newPeerIdx = rasLinkCalculatePeer(link, myPeerIdx, /*isFallback*/false);
  linkConn->conn = (newPeerIdx != -1 ? rasConnFind(&rasPeers[newPeerIdx].addr) : nullptr);
  linkConn->external = false;

  if (linkConn->conn == nullptr) {
    if (linkConn->peerIdx != -1) {
      // We try to initiate primary connections from the side with a lower address (and thus an earlier peer index)
      // to avoid races and the creation of duplicate connections.
      INFO(NCCL_RAS, "RAS link %d: %s primary connection with %s",
           link->direction, (myPeerIdx < linkConn->peerIdx ? "opening new" : "calculated deferred"),
           ncclSocketToString(&rasPeers[linkConn->peerIdx].addr, rasLine));
      if (myPeerIdx < linkConn->peerIdx) {
        NCCLCHECK(rasConnCreate(&rasPeers[linkConn->peerIdx].addr, &linkConn->conn));
      }
      else { // If we didn't initiate the connection, start the timeout.
        link->lastUpdatePeersTime = clockNano();
      }
    } // if (linkConn->peerIdx != -1)
  } else { // linkConn->conn
    INFO(NCCL_RAS, "RAS link %d: calculated existing primary connection with %s",
         link->direction, ncclSocketToString(&rasPeers[linkConn->peerIdx].addr, rasLine));
  } // linkConn->conn

  if (linkConn->conn && linkConn->conn->experiencingDelays) {
    INFO(NCCL_RAS, "RAS connection experiencingDelays %d, startRetryTime %.2fs, socket status %d",
         linkConn->conn->experiencingDelays, (clockNano()-linkConn->conn->startRetryTime)/1e9,
         (linkConn->conn->sock ? linkConn->conn->sock->status : - 1));
    NCCLCHECK(rasLinkAddFallback(link, linkConn->conn));
  }

  return ncclSuccess;
}

// Calculates the index of the peer on the RAS network.  Can also be used to calculate the index of the next fallback
// peer.
// In the simplest case we want to try the "next closest" fallback, although we still need to check for and skip
// any dead peers.
// For fallbacks to fallbacks, we also apply a more pessimistic policy.  We skip all the remaining RAS threads that
// are on the same node as the previous fallback (unless it's the same node that we're running on or we have strong
// indications that the node is up).  We do that to avoid having to excessively wait iterating through, say, 8
// processes when a whole node might be down.
int rasLinkCalculatePeer(const struct rasLink* link, int peerIdx, bool isFallback) {
  int newPeerIdx = (peerIdx + link->direction + nRasPeers) % nRasPeers;
  do {
    if (isFallback && !ncclSocketsSameNode(&rasPeers[peerIdx].addr, &rasNetListeningSocket.addr)) {
      // peerIdx is a fallback and it is not running on the same node as us.
      int tryPeerIdx = newPeerIdx;
      struct rasConnection* tryConn = nullptr;

      // Try to skip the remaining peers on the same node as peerIdx.  We may end up skipping over some peers that
      // are alive, which is fine -- they will still have connectivity with the rest of the RAS network, just a
      // little suboptimal one.
      while (ncclSocketsSameNode(&rasPeers[tryPeerIdx].addr, &rasPeers[peerIdx].addr)) {
        if (!rasPeerIsDead(&rasPeers[tryPeerIdx].addr)) {
          tryConn = rasConnFind(&rasPeers[tryPeerIdx].addr);
          if (tryConn) {
            // Check if the connection is fully established and operational, i.e., if the underlying socket
            // is ready and there's been recent communication on it.
            if (tryConn->sock && tryConn->sock->status == RAS_SOCK_READY && !tryConn->experiencingDelays) {
              // We convinced ourselves that the node is not down.  We don't adjust newPeerIdx in
              // this case.  This is the only case when tryConnIdx != -1 after this loop.
              break;
            }
          } // if (tryConn)
        } // if (!rasPeerIsDead(&rasPeers[tryPeerIdx].addr))

        tryConn = nullptr;
        tryPeerIdx = (tryPeerIdx + link->direction + nRasPeers) % nRasPeers;
        if (tryPeerIdx == myPeerIdx)
          break;
      }

      if (tryConn == nullptr)
        newPeerIdx = tryPeerIdx;
      if (tryPeerIdx == myPeerIdx)
        break;
    } // if (isFallback && !ncclSocketsSameNode(&rasPeers[peerIdx].addr, &rasNetListeningSocket.addr))

    if (rasPeerIsDead(&rasPeers[newPeerIdx].addr)) {
      newPeerIdx = (newPeerIdx + nRasPeers + link->direction) % nRasPeers;
    }
    else
      break;
  } while (newPeerIdx != myPeerIdx);

  return (newPeerIdx != myPeerIdx ? newPeerIdx : -1);
}


//////////////////////////////////////////////////////
// Functions related to the handling of dead peers. //
//////////////////////////////////////////////////////

// Marks a peer as dead in the local rasDeadPeers array.  Any propagation, reconfiguration, etc., needs to be
// handled outside of this function.
ncclResult_t rasPeerDeclareDead(const union ncclSocketAddress* addr) {
  union ncclSocketAddress* deadAddr;

  if (!rasPeerIsDead(addr)) {
    NCCLCHECK(getNewDeadEntry(&deadAddr));
    memcpy(deadAddr, addr, sizeof(*deadAddr));
    qsort(rasDeadPeers, nRasDeadPeers, sizeof(*rasDeadPeers), &ncclSocketsCompare);

    rasDeadPeersHash = getHash((const char*)rasDeadPeers, nRasDeadPeers*sizeof(*rasDeadPeers));

    INFO(NCCL_RAS, "RAS declaring peer %s as DEAD; rasDeadPeersHash 0x%lx",
         ncclSocketToString(addr, rasLine), rasDeadPeersHash);
  }
  return ncclSuccess;
}

// Invoked when an incoming RAS_MSG_PEERSUPDATE includes info on dead peers.  Updates the rasDeadPeers array.
// Any propagation needs to be handled outside of this function, though it *does* disconnect any connections
// with the newly dead peers.
// On return, nUpdatePeers contains the number of newly added dead entries.
static ncclResult_t rasDeadPeersUpdate(union ncclSocketAddress* updatePeers, int* nUpdatePeers) {
  static union ncclSocketAddress* newPeers = nullptr;
  static union ncclSocketAddress* oldPeers;

  if (*nUpdatePeers == 0)
    return ncclSuccess;

  // Pessimistically estimate the new size of rasDeadPeers.
  int nNewPeers = nRasDeadPeers + *nUpdatePeers;
  if (nNewPeers > rasDeadPeersSize) {
    nNewPeers = ROUNDUP(nNewPeers, RAS_INCREMENT);

    NCCLCHECK(ncclCalloc(&newPeers, nNewPeers));
    oldPeers = rasDeadPeers;
  } else {
    // We don't need to allocate a new array in this case.  We just shift the existing content to the end of the
    // array to make room in the front for merging.
    oldPeers = rasDeadPeers+(rasDeadPeersSize-nRasDeadPeers);
    memmove(oldPeers, rasDeadPeers, nRasDeadPeers*sizeof(*rasDeadPeers));
    newPeers = rasDeadPeers;
  }

  // Merge updatePeers with oldPeers into newPeers.
  int oldPeersIdx, updatePeersIdx, newPeersIdx;
  for (oldPeersIdx = updatePeersIdx = newPeersIdx = 0; oldPeersIdx < nRasDeadPeers || updatePeersIdx < *nUpdatePeers;) {
    int cmp;
    if (oldPeersIdx < nRasDeadPeers && updatePeersIdx < *nUpdatePeers) {
      cmp = ncclSocketsCompare(oldPeers+oldPeersIdx, updatePeers+updatePeersIdx);
    } else {
      cmp = (oldPeersIdx < nRasDeadPeers ? -1 : 1);
    }

    memmove(newPeers+newPeersIdx++, (cmp <= 0 ? oldPeers+oldPeersIdx : updatePeers+updatePeersIdx), sizeof(*newPeers));
    if (cmp <= 0)
      oldPeersIdx++;
    if (cmp > 0) {
      rasConnDisconnect(updatePeers+updatePeersIdx);
    }
    if (cmp >= 0)
      updatePeersIdx++;
  }
  *nUpdatePeers = newPeersIdx - nRasDeadPeers;
  nRasDeadPeers = newPeersIdx;

  if (newPeers != rasDeadPeers) {
    free(rasDeadPeers);
    rasDeadPeers = newPeers;
    rasDeadPeersSize = nNewPeers;
  }

  rasDeadPeersHash = getHash((const char*)rasDeadPeers, nRasDeadPeers*sizeof(*rasDeadPeers));

  return ncclSuccess;
}

// Returns the index of the first available entry in the rasDeadPeers array, enlarging the array if necessary.
static ncclResult_t getNewDeadEntry(union ncclSocketAddress** pAddr) {
  if (nRasDeadPeers == rasDeadPeersSize) {
    NCCLCHECK(ncclRealloc(&rasDeadPeers, rasDeadPeersSize, rasDeadPeersSize+RAS_INCREMENT));
    rasDeadPeersSize += RAS_INCREMENT;
  }

  *pAddr = rasDeadPeers+(nRasDeadPeers++);
  return ncclSuccess;
}

// Checks whether a peer is dead by looking it up in the rasDeadPeers array.
bool rasPeerIsDead(const union ncclSocketAddress* addr) {
  return (rasDeadPeers != nullptr &&
          bsearch(addr, rasDeadPeers, nRasDeadPeers, sizeof(*rasDeadPeers), ncclSocketsCompare) != nullptr);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Auxiliary functions -- primarily sorting/searching callbacks, plus some debug output support. //
///////////////////////////////////////////////////////////////////////////////////////////////////

// Searching callback for struct rasRankInit.  Compares the ncclSocketAddress key against a rasRankInit element.
static int rasAddrRankInitCompare(const void* k, const void* e) {
  const union ncclSocketAddress* key = (const union ncclSocketAddress*)k;
  const struct rasRankInit* elem = (const struct rasRankInit*)e;

  return ncclSocketsCompare(key, &elem->addr);
}

// Searching callback for struct rasPeerInfo.  Compares the ncclSocketAddress key against a rasPeerInfo element.
static int rasAddrPeerInfoCompare(const void* k, const void* e) {
  const union ncclSocketAddress* key = (const union ncclSocketAddress*)k;
  const struct rasPeerInfo* elem = (const struct rasPeerInfo*)e;

  return ncclSocketsCompare(key, &elem->addr);
}

// Sorting callback for struct rasRankInit. addr is the primary key; cudaDev is secondary.
static int rasRanksCompare(const void* e1, const void* e2) {
  const struct rasRankInit* r1 = (const struct rasRankInit*)e1;
  const struct rasRankInit* r2 = (const struct rasRankInit*)e2;
  int cmp = ncclSocketsCompare(&r1->addr, &r2->addr);
  if (cmp == 0) {
    if (r1->addr.sa.sa_family == 0) // Bail out in case of empty addresses...
      return 0;
    assert(r1->pid == r2->pid);
    cmp = (r1->cudaDev < r2->cudaDev ? -1 : (r1->cudaDev > r2->cudaDev ? 1 : 0));
    assert(cmp != 0); // There should be no complete duplicates within the rank array.
  }
  return cmp;
}

// Sorting callback for ncclSocketAddress.  We want to sort by the address family (IPv4 first), then the address,
// then port.  Unfortunately, that's not the order of how they are laid out in memory, so one big memcmp won't do.
// memcmp is still useful though for individual elements in the network byte order.
int ncclSocketsCompare(const void* p1, const void* p2) {
  const union ncclSocketAddress* a1 = (const union ncclSocketAddress*)p1;
  const union ncclSocketAddress* a2 = (const union ncclSocketAddress*)p2;
  // AF_INET (2) is less than AF_INET6 (10).
  int family = a1->sa.sa_family;
  if (family != a2->sa.sa_family) {
    if (family > 0 && a2->sa.sa_family > 0)
      return (family < a2->sa.sa_family ? -1 : 1);
    else // Put empty addresses at the end (not that it matters...).
      return (family > 0 ? -1 : 1);
  }

  int cmp;
  if (family == AF_INET) {
    if ((cmp = memcmp(&a1->sin.sin_addr, &a2->sin.sin_addr, sizeof(a1->sin.sin_addr))) == 0) {
      cmp = memcmp(&a1->sin.sin_port, &a2->sin.sin_port, sizeof(a1->sin.sin_port));
    }
  }
  else if (family == AF_INET6) {
    if ((cmp = memcmp(&a1->sin6.sin6_addr, &a2->sin6.sin6_addr, sizeof(a1->sin6.sin6_addr))) == 0) {
      cmp = memcmp(&a1->sin6.sin6_port, &a2->sin6.sin6_port, sizeof(a1->sin6.sin6_port));
    }
  } else {
    // The only remaining valid case are empty addresses.
    assert(family == 0);
    cmp = 0; // Two empty addresses are equal...
  }

  return cmp;
}

// Returns true if two socket addresses are from the same node (actually, the same network interface on one node).
bool ncclSocketsSameNode(const union ncclSocketAddress* a1, const union ncclSocketAddress* a2) {
  // AF_INET (2) is less than AF_INET6 (10).
  int family = a1->sa.sa_family;
  if (family != a2->sa.sa_family)
    return false;

  if (family == AF_INET)
    return (memcmp(&a1->sin.sin_addr, &a2->sin.sin_addr, sizeof(a1->sin.sin_addr)) == 0);
  else if (family == AF_INET6)
    return (memcmp(&a1->sin6.sin6_addr, &a2->sin6.sin6_addr, sizeof(a1->sin6.sin6_addr)) == 0);
  else
    return true; // Two empty addresses are equal...
}

// Debug output routine: dumps the rasPeers array.
static void rasPeersDump() {
  for (int p = 0; p < nRasPeers; p++) {
    const struct rasPeerInfo* peer = rasPeers+p;
    INFO(NCCL_RAS, "RAS peer %d: %s%s", p, rasPeerDump(peer, rasLine, sizeof(rasLine)),
         (p == myPeerIdx ? " [this process]" : ""));
  }
  if (nRasPeers > 0)
    INFO(NCCL_RAS, "RAS peersHash 0x%lx", rasPeersHash);
}

// Debug output routine: dumps the rasDeadPeers array.
static void rasDeadPeersDump() {
  for (int p = 0; p < nRasDeadPeers; p++) {
    int deadPeerIdx = rasPeerFind(rasDeadPeers+p);
    INFO(NCCL_RAS, "RAS dead peer %d: %s", p,
         (deadPeerIdx >= 0 ? rasPeerDump(rasPeers+deadPeerIdx, rasLine, sizeof(rasLine)) :
          ncclSocketToString(rasDeadPeers+p, rasLine)));
  }
  if (nRasDeadPeers > 0)
    INFO(NCCL_RAS, "RAS deadPeersHash 0x%lx", rasDeadPeersHash);
}

// Debug output routine: dumps part of an individual element from the rasPeers array.
static char* rasPeerDump(const struct rasPeerInfo* peer, char* result, size_t nres) {
  char line[SOCKET_NAME_MAXLEN+1], line2[1024];
  snprintf(result, nres, "socket %s, pid %d, GPU%s %s", ncclSocketToString(&peer->addr, line), peer->pid,
           (__builtin_popcountll(peer->cudaDevs) > 1 ? "s" : ""),
           rasGpuDevsToString(peer->cudaDevs, peer->nvmlDevs, line2, sizeof(line2)));
  return result;
}

// Invoked during RAS termination to release all the allocated resources.
void rasPeersTerminate() {
  free(rasPeers);
  rasPeers = nullptr;
  nRasPeers = 0;
  rasPeersHash = 0;
  myPeerIdx = -1;

  free(rasDeadPeers);
  rasDeadPeers = nullptr;
  nRasDeadPeers = rasDeadPeersSize = 0;
  rasDeadPeersHash = 0;
}
