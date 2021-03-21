#include <iostream>


int main(int argc, char **argv) {
    if (argc != 2)
    {
        std::cerr << "You must run this program as: bradley <threads count> <path to image>";
        return 1;
    }
    std::cout << "Bradley algorithm on image " << argv[1] << std::endl;

    return 0;
}
