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
