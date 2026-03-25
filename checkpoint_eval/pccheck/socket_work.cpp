#include "socket_work.h"
#include <cerrno>

namespace {
constexpr int kConnectRetryMs = 200;
constexpr int kConnectMaxRetries = 600; // ~120s
}

void setup_rank0_socket(const int port, int* server_fd, struct sockaddr_in* address, int N, std::vector<int> &client_sockets) {

    printf("Inside setup_rank0_socket\n");
    *server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (*server_fd < 0) {
        perror("socket");
        exit(1);
    }

    int opt = 1;
    if (setsockopt(*server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
        perror("setsockopt(SO_REUSEADDR|SO_REUSEPORT)");
        close(*server_fd);
        exit(1);
    }

    address->sin_family = AF_INET;
    address->sin_addr.s_addr = INADDR_ANY;
    address->sin_port = htons(port);
    int addrlen = sizeof(*address);

    // Bind the socket
    if (bind(*server_fd, (struct sockaddr *)address, sizeof(*address)) < 0) {
        perror("bind");
        close(*server_fd);
        exit(1);
    }
    if (listen(*server_fd, N) < 0) {
        perror("listen");
        close(*server_fd);
        exit(1);
    }

    // Wait for all nodes to connect
    for (int i=0; i<N; i++) {
        printf("Get new connection for node %d\n", i);
        int new_socket = accept(*server_fd, (struct sockaddr *)address, (socklen_t *)&addrlen);
        if (new_socket < 0) {
            perror("accept");
            close(*server_fd);
            exit(1);
        }
        client_sockets.push_back(new_socket);
        printf("Node %d connected!\n", i);
    }

}


void setup_other_socket(int* sock, struct sockaddr_in* serv_addr, const std::string &server_ip, int port) {
    serv_addr->sin_family = AF_INET;
    serv_addr->sin_port = htons(port);
    if (inet_pton(AF_INET, server_ip.c_str(), &(serv_addr->sin_addr)) <= 0) {
        fprintf(stderr, "[ERROR] invalid PCCHECK_COORDINATOR ip: %s\n", server_ip.c_str());
        exit(1);
    }

    for (int attempt = 1; attempt <= kConnectMaxRetries; ++attempt) {
        *sock = socket(AF_INET, SOCK_STREAM, 0);
        if (*sock < 0) {
            perror("socket");
            usleep(kConnectRetryMs * 1000);
            continue;
        }

        if (connect(*sock, (struct sockaddr *)serv_addr, sizeof(*serv_addr)) == 0) {
            printf("Connected!\n");
            return;
        }

        int err = errno;
        close(*sock);

        // server may not be up yet, retry quietly
        if (err == ECONNREFUSED || err == ETIMEDOUT || err == EHOSTUNREACH || err == ENETUNREACH) {
            if (attempt % 25 == 0) {
                fprintf(stderr,
                        "[WARN] connect retry %d/%d to %s:%d (errno=%d)\n",
                        attempt, kConnectMaxRetries, server_ip.c_str(), port, err);
            }
            usleep(kConnectRetryMs * 1000);
            continue;
        }

        fprintf(stderr,
                "[ERROR] connect failed to %s:%d (errno=%d)\n",
                server_ip.c_str(), port, err);
        exit(1);
    }

    fprintf(stderr,
            "[ERROR] timeout connecting to coordinator %s:%d after %d retries\n",
            server_ip.c_str(), port, kConnectMaxRetries);
    exit(1);

}

void wait_to_receive(std::vector<int>& client_sockets, int N) {

    for (auto sock: client_sockets) {
    int* iter = (int*)malloc(sizeof(int));
    ssize_t r1 = read(sock, iter, 4);
    (void)r1;
    }

    for (int sock : client_sockets) {
        int val = 1;
        send(sock, &val, 4, 0);
    }

}

void send_and_wait(int* socket, int counter) {
    send(*socket, &counter, 4, 0);
    int* val = (int*)malloc(sizeof(int));
    ssize_t r2 = read(*socket, val, 4);
    (void)r2;
}

void close_rank0_socket(std::vector<int>& client_sockets, int* server_fd) {

    for (int sock : client_sockets) {
        close(sock);
    }
    close(*server_fd);
}

void close_other_socket(int* sock) {
    close(*sock);

}
