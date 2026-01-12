#include <iostream>

int main() {
  auto Print = [](const std::string& msg){std::cout << msg << std::endl;};
  Print("Hello, world!");
}
