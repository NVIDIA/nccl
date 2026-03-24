#ifndef INSPECTOR_INSPECTOR_RING_H_
#define INSPECTOR_INSPECTOR_RING_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>

// Forward declaration — inspectorResult_t has a fixed underlying type so it
// can be declared here without pulling in inspector.h (which would be circular,
// since inspector.h includes this header for inspectorCompletedRing).
enum inspectorResult_t : int;

// Generic fixed-size ring buffer (used for both coll and P2P completed entries).
// Defined here rather than in inspector.h to keep ring logic self-contained;
// inspector.h includes this header so inspectorCommInfo can embed the struct.
//
// Uses Sentinel Approach (never-full) - one sentinel slot is allocated beyond 'size' so
// that empty (head==tail) and full ((tail+1)%(size+1)==head) are always
// distinguishable without a separate count field.
// Empty : head == tail
// Full  : (tail + 1) % (size + 1) == head  — overwrite oldest on enqueue
struct inspectorCompletedRing {
  void* entries{nullptr};
  size_t entrySize{0};  // size of each entry in bytes
  uint32_t size{0};     // user-visible capacity (size+1 slots are allocated)
  uint32_t head{0};     // index of oldest element
  uint32_t tail{0};     // index where next element will be written
};

/*
 * Description:
 *   Returns a pointer to the entry at index idx in the ring's backing store.
 *
 *   Because entries is void* (the ring is shared by coll and P2P), array
 *   subscript notation is not available. This helper centralises the byte-offset
 *   arithmetic (idx * entrySize) so no call site needs to spell it out.
 *
 * Parameters:
 *   ring - ring buffer whose backing store is accessed.
 *   idx  - slot index (must be < ring->size).
 *
 * Return:
 *   void* - pointer to the entry at the given slot.
 */
static inline void* ringSlot(const struct inspectorCompletedRing* ring, uint32_t idx) {
  return (char*)ring->entries + idx * ring->entrySize;
}

// Returns true when the ring contains at least one entry.
static inline bool inspectorRingNonEmpty(const struct inspectorCompletedRing* ring) {
  return ring->head != ring->tail;
}

// Generic ring operations
inspectorResult_t inspectorRingInit(struct inspectorCompletedRing* ring,
                                    uint32_t size,
                                    size_t entrySize);
void inspectorRingFinalize(struct inspectorCompletedRing* ring);
inspectorResult_t inspectorRingEnqueue(struct inspectorCompletedRing* ring,
                                       const void* entry);

/*
 * Description:
 *   Drains all entries from the ring buffer into a typed scratch vector.
 *
 *   Templated so it works for any entry type without requiring this header
 *   to include inspector.h (which would be circular). The caller's vector
 *   type T must match the entry type the ring was initialised with.
 *
 *   Non-dependent names in the template body cannot use inspectorSuccess /
 *   inspectorMemoryError by name (two-phase lookup resolves them at definition
 *   time, before inspector.h is seen). Integer literals are used instead, with
 *   the enum's fixed underlying type making the conversion well-defined.
 *
 * Thread Safety:
 *   Not thread-safe (caller must provide synchronization).
 *
 * Parameters:
 *   ring    - ring buffer to drain.
 *   scratch - output vector; cleared and filled with copies of ring entries.
 *
 * Return:
 *   inspectorResult_t{0} (inspectorSuccess)     on success.
 *   inspectorResult_t{2} (inspectorMemoryError) if ring is null.
 */
template<typename T>
inspectorResult_t inspectorRingDrain(struct inspectorCompletedRing* ring,
                                     std::vector<T>& scratch) {
  scratch.clear();
  if (!ring) return static_cast<inspectorResult_t>(2); // inspectorMemoryError
  if (ring->head != ring->tail && ring->size > 0 && ring->entries) {
    uint32_t bufSize = ring->size + 1;
    uint32_t count   = (ring->tail + bufSize - ring->head) % bufSize;
    for (uint32_t i = 0; i < count; i++) {
      uint32_t idx = (ring->head + i) % bufSize;
      scratch.push_back(*static_cast<T*>(ringSlot(ring, idx)));
    }
    ring->head = 0;
    ring->tail = 0;
  }
  return static_cast<inspectorResult_t>(0); // inspectorSuccess
}

#endif  // INSPECTOR_INSPECTOR_RING_H_
