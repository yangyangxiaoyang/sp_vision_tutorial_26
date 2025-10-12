#include <fmt/core.h>

int main() {
  fmt::print("Part1: fmt test \n");
  fmt::print("Hello, fmt!\n");  

  std::string name = "cfy";
  int age = 19;
  double score = 99.5;
  std::string info = fmt::format(
    "User Info: Name={}, Age={}, Score={:.1f}", 
    name, age, score                             
  );
  fmt::print("{}\n", info);  

  return 0;
}
