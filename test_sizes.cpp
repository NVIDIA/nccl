#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdio.h>

// Connection state machine
enum ncclSocketCommState
{
    ncclSocketCommStateStart = 0,
    ncclSocketCommStateConnect = 1,
    ncclSocketCommStateAccept = 2,
    ncclSocketCommStateReady = 3
};

// Forward declaration
struct ncclSocketComm;

// Staging structure for async connect/accept
struct ncclSocketCommStage
{
    enum ncclSocketCommState state;
    struct ncclSocketComm *comm;
    SOCKET sock;
};

// Socket handle structure
struct ncclSocketHandle
{
    unsigned long long magic;
    struct sockaddr_in connectAddr;
    struct ncclSocketCommStage stage; 
};

int main() {
    printf("SOCKET size: %zu\n", sizeof(SOCKET));
    printf("void* size: %zu\n", sizeof(void*));
    printf("ncclSocketCommStage size: %zu\n", sizeof(struct ncclSocketCommStage));
    printf("sockaddr_in size: %zu\n", sizeof(struct sockaddr_in));
    printf("ncclSocketHandle size: %zu\n", sizeof(struct ncclSocketHandle));
    printf("NCCL_NET_HANDLE_MAXSIZE is 128\n");
    return 0;
}
