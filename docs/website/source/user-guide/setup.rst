.. _setup-label:

#####
Setup
#####

NCCL is a communication library providing optimized GPU-to-GPU communication for high-performance applications.
It is not, like MPI, providing a parallel environment including a process launcher and manager. NCCL relies therefore
on the application's process management system and CPU-side communication system for its own bootstrap.

By default, NCCL does not encrypt network communication. It is therefore the responsibility of the user to ensure NCCL
operates over a secure network, both for bootstrap (controlled by :ref:`NCCL_SOCKET_IFNAME`) and for high-speed
communication. An OpenSSL-enabled build can encrypt NCCL-owned TCP traffic carried by ``ncclSocket`` when an encryption
key is configured. Non-Socket transports remain outside that protection boundary.

Building with TLS support
=========================

TLS support requires OpenSSL 3 headers and shared libraries. Install the OpenSSL development package through the system
package manager so the headers are available during the build and NCCL can load the system libraries at run time.

For Make, select OpenSSL when building NCCL:

.. code-block:: shell

   make TLS_BACKEND=OPENSSL3 src.build

For CMake, select the same backend during configuration:

.. code-block:: shell

   cmake -S . -B build-cmake -DTLS_BACKEND=OPENSSL3
   cmake --build build-cmake -j

An unset or empty ``TLS_BACKEND`` builds NCCL without TLS support or an OpenSSL dependency. Other nonempty values are
rejected. Both build systems require OpenSSL 3 when the backend is selected and fail if its headers or libraries cannot
be found.

Configuring TCP encryption
==========================

Applications that require encrypted NCCL-owned TCP sockets must configure encryption with :c:func:`ncclSetEncryption`.
Every process in the job should configure the same high-entropy pre-shared key (PSK) before calling
:c:func:`ncclGetUniqueId` or otherwise starting NCCL networking:

.. code-block:: c

   ncclEncryptionConfig_t encryption = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
   encryption.mode = NCCL_ENCRYPTION_MODE_PSK;
   encryption.psk = jobPsk; /* Null-terminated string containing at least 32 bytes. */
   NCCLCHECK(ncclSetEncryption(&encryption));

   if (rank == 0) NCCLCHECK(ncclGetUniqueId(&uniqueId));
   MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
   NCCLCHECK(ncclCommInitRank(&comm, nranks, uniqueId, rank));

NCCL uses the configured PSK to authenticate TLS 1.3 handshakes and derive fresh traffic keys for each NCCL-owned TCP
connection. If encryption is configured but NCCL was built without TLS support, cannot load a compatible OpenSSL 3
runtime, or receives invalid key material, NCCL returns an error rather than continuing without encryption.

The PSK is process-global state. Applications should configure it once per job, before any communicator or NCCL unique
ID is created, and keep it unchanged until the job ends. Changing encryption configuration while NCCL networking or RAS
peer connections are active is not supported and may prevent peers using different PSKs from reconnecting.

Encryption covers traffic carried through NCCL's built-in socket layer, including NCCL bootstrap and coordination over
TCP, the built-in Socket network transport, and NCCL services that use ``ncclSocket``. It does not encrypt transports
that bypass NCCL sockets, such as IB/RDMA, NVLink, shared memory, GPU-side data paths, or network/bootstrap plugins that
provide their own transport. Applications remain responsible for securing their process launcher, PSK distribution, and
out-of-band exchange of NCCL unique IDs.
