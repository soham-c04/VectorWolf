#pragma once
#include <iostream>
#include <vector>
#include <string>
using namespace std;
using D = double;

// For printing boxes
#define VERT         (char)186
#define HORIZ        (char)205
#define TOP_LEFT     (char)201
#define TOP_RIGHT    (char)187
#define BOTTOM_LEFT  (char)200
#define BOTTOM_RIGHT (char)188
#define SCREEN_WIDTH 130

string lower_case(const string &s);

void print(string &line, int width = SCREEN_WIDTH);
void print_top(int width = SCREEN_WIDTH);
void print_bottom(int width = SCREEN_WIDTH);
void print_header(string line);
void print(const vector<D> &vec);
void print(const vector<vector<D>> &vec);

void shape(vector<vector<D>> &M);

vector<vector<D>> transpose(vector<vector<D>> &M);
vector<vector<D>> multiply(vector<vector<D>> &a, vector<vector<D>> &b);
vector<vector<D>> multiply(vector<vector<D>> M, int c);
vector<D> multiply(vector<D> M, int c);
vector<vector<D>> hadamard_product(vector<vector<D>> a, vector<vector<D>> &b);
vector<D> hadamard_product(vector<D> a, vector<D> &b);
