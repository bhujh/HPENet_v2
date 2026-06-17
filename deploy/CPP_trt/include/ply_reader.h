#pragma once
#include "types.h"
#include <string>

class PlyReader {
public:
    static PointCloud load(const std::string& path);
};
