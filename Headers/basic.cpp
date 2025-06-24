#pragma once
#include "basic.h"

string lower_case(const string &s){
	string low="";
	for(char c:s) low.push_back(c|32);
	return low;
}

void print(string &line, int width){
	int l = line.size();
	cout<<VERT<<"  "<<line;
	for(int p=0;p<width-l-2;p++) cout<<" ";
	cout<<VERT<<"\n";
	line.clear();
}

void print_top(int width){
	cout<<TOP_LEFT;
	for(int c=0;c<width;c++) cout<<HORIZ;
	cout<<TOP_RIGHT<<"\n";
	if(width == SCREEN_WIDTH){
		string line;
		print(line = "");
	}
}

void print_bottom(int width){
	if(width == SCREEN_WIDTH){
		string line;
		print(line = "");
	}
	cout<<BOTTOM_LEFT;
	for(int c=0;c<width;c++) cout<<HORIZ;
	cout<<BOTTOM_RIGHT<<endl;
}

void print_header(string line){
	int l = line.size();
	print_top(l + 4);
	print(line,l + 4);
	print_bottom(l + 4);
}

void print(const vector<D> &vec){
	string line = "      ";
	for(D a:vec) line += to_string(a) + ' ';
	print(line);
}

void print(const vector<vector<D>> &vec){
	for(vector<D> v:vec) print(v);
}

void shape(vector<vector<D>> &M){
	cout<<"\n(";
	if(M.empty()) cout<<",";
	else{
		cout<<M.size()<<",";
		if(M[0].size() != 0) cout<<M[0].size();
	}
	cout<<")"<<endl;
}

vector<vector<D>> transpose(vector<vector<D>> &M){
	int n=M.size(),m=M[0].size();
	vector<vector<D>> M_t(m,vector<D>(n));
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			M_t[j][i]=M[i][j];

	return M_t;
}

vector<vector<D>> multiply(vector<vector<D>> &a, vector<vector<D>> &b){
	int p = a.size(), q = a[0].size(), r = b[0].size();
	vector<vector<D>> ans(p,vector<D>(r));

	if(q != b.size()) cout<<"\nDimension mismatch.\n";
	else{
		for(int i=0;i<p;i++){
			for(int j=0;j<r;j++){
				D ans_ij = 0;
				for(int k=0;k<q;k++) ans_ij += a[i][k]*b[k][j];
				ans[i][j] = ans_ij;
			}
		}
	}
	return ans;
}
