.. py:currentmodule:: nccl.core

***********************************
Point-to-Point and Signal Methods
***********************************

Methods on :py:class:`Communicator` for point-to-point and signal/wait
operations. See :doc:`../../api/p2p` for the corresponding C API.

send
----
.. automethod:: Communicator.send

recv
----
.. automethod:: Communicator.recv

signal
------
.. automethod:: Communicator.signal

wait_signal
-----------
.. automethod:: Communicator.wait_signal

put_signal
----------
.. automethod:: Communicator.put_signal

WaitSignalDesc
--------------
.. autoclass:: WaitSignalDesc
