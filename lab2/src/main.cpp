#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

int main() {
  // X - random number between 0 and 1
  // Y - function of X (e.g. "2X + 1 + eps" in example.data)
  std::vector<double> X, Y;

  // DATA GETTING
  std::cout << "Enter file with X & Y data:\n> ";
  std::string path;
  std::getline(std::cin, path);
  std::ifstream file(path);

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string value;
    double x, y;
    while (std::getline(iss, value, '\t')) { // reading X, tab is separator
      std::replace(value.begin(), value.end(), ',', '.'); // replace , to .
      if (!(std::istringstream(value) >> x)) {
        std::cerr << "Error while reading X from file\n";
        return 1;
      }
      if (std::getline(iss, value, '\t')) { // reading Y
        std::replace(value.begin(), value.end(), ',', '.');
        if (!(std::istringstream(value) >> y)) {
          std::cerr << "Error while reading Y from file\n";
          return 1;
        }
      }
      X.push_back(x);
      Y.push_back(y);
    }
  }

  // DATA PROCESSING
  // mean of X and Y
  double meanX = 0.0, meanY = 0.0;
  for (size_t i = 0; i < X.size(); i++) {
    meanX += X[i];
    meanY += Y[i];
  }
  meanX /= X.size();
  meanY /= Y.size();

  // calculate coefficients
  double numerator = 0.0, denominator = 0.0;
  for (size_t i = 0; i < X.size(); i++) {
    numerator += (X[i] - meanX) * (Y[i] - meanY);
    denominator += (X[i] - meanX) * (X[i] - meanX);
  }
  double A = numerator / denominator;
  double B = meanY - A * meanX;

  std::cout << "Y = " << A << "X + " << B << '\n';
  return 0;
}