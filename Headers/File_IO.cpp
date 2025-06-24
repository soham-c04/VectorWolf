#include "File_IO.h"
#include "basic.cpp"

// Keyword for read_csv()

map<string,map<string,D>> dummy_replace;
vector<string> null_values = {""};

void write_csv(const string path, vector<D> data){
	cout<<endl;
	print_header("write_csv");
	
	print_top();
	ofstream file(path);
	
	if(!file.is_open()){
		string line = "Unable to open file: " + path;
		print(line);
		return;
	}
	
	int m = data.size();
    for(int i=0; i<m; i++){
        file << data[i];
        if(i != data.size() - 1)
			file << ",";
    }

    file << endl;
    file.close();
    
    print_bottom();
}
