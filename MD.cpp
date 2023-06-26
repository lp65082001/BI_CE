#include <iostream>
#include <vector>
#include <cmath>
#include <random>

const double L = 10.0; // 容器邊長
const int N = 100; // 分子數量
const double dt = 0.01; // 時間間隔
const int steps = 1000; // 模擬步數

struct Atom {
    double x, y, z;
    double vx, vy, vz;
};

double randomDouble(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

void initializeAtoms(std::vector<Atom>& atoms) {
    for (int i = 0; i < N; i++) {
        Atom atom;
        atom.x = randomDouble(0, L);
        atom.y = randomDouble(0, L);
        atom.z = randomDouble(0, L);
        atom.vx = randomDouble(-1, 1);
        atom.vy = randomDouble(-1, 1);
        atom.vz = randomDouble(-1, 1);
        atoms.push_back(atom);
    }
}

void updatePositions(std::vector<Atom>& atoms) {
    for (int i = 0; i < N; i++) {
        atoms[i].x += atoms[i].vx * dt;
        atoms[i].y += atoms[i].vy * dt;
        atoms[i].z += atoms[i].vz * dt;
        
        // 碰壁檢測
        if (atoms[i].x < 0 || atoms[i].x > L) {
            atoms[i].vx *= -1;
        }
        if (atoms[i].y < 0 || atoms[i].y > L) {
            atoms[i].vy *= -1;
        }
        if (atoms[i].z < 0 || atoms[i].z > L) {
            atoms[i].vz *= -1;
        }
    }
}

void simulate() {
    std::vector<Atom> atoms;
    initializeAtoms(atoms);
    
    for (int step = 0; step < steps; step++) {
        updatePositions(atoms);
        
        // 輸出分子位置
        for (int i = 0; i < N; i++) {
            std::cout << "Atom " << i+1 << ": ";
            std::cout << atoms[i].x << " " << atoms[i].y << " " << atoms[i].z << std::endl;
        }
        std::cout << std::endl;
    }
}

int main() {
    simulate();
    
    return 0;
}