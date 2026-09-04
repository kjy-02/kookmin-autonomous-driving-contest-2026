#include <cerrno>
#include <cstdio>
#include <cstring>
#include <vector>

#include <unistd.h>

int main(int argc, char **argv) {
  // Replace this process so ROS launch arguments and signals are delivered
  // directly to the Python traffic-vision node.  Building this small launcher
  // avoids depending on executable permission bits of copied Python scripts.
  std::vector<char *> python_arguments;
  python_arguments.reserve(static_cast<std::size_t>(argc) + 4U);
  python_arguments.push_back(const_cast<char *>("python3"));
  python_arguments.push_back(const_cast<char *>("-m"));
  python_arguments.push_back(const_cast<char *>("final_drive_vision.node"));
  for (int index = 1; index < argc; ++index) {
    python_arguments.push_back(argv[index]);
  }
  python_arguments.push_back(nullptr);

  execvp(python_arguments.front(), python_arguments.data());
  std::fprintf(
      stderr,
      "failed to start final_drive_vision.node with python3: %s\n",
      std::strerror(errno));
  return 127;
}
