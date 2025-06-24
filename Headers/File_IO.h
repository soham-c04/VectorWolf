#include <cmath>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>

// Keyword arguments for read_csv()

/*
dummy_replace is can be used to format .csv while reading
by replacing certain strings of a certain header by a numeric.
E.g before using read_csv() do:
dummy_replace["Gender"]["male"] = 1;
dummy_replace["Gender"]["female"] = 0;
read_csv()

This will replace male under Gender column with 1 and female by 0.
If this is not done, "Gender" column will be ommitted to be used further in the model.
NOTE:- Once a column is included in dummy_replace all its occurences (even if it is numeric) should be mentioned in dummy_replace.
	   Otherwise, it is set to NAN, and can be later identified by - isnan(data[i][j])

dummy_replace is cleared() after each read_csv use.
*/

extern map<string,map<string,D>> dummy_replace;
extern vector<string> null_values; // Strings which are treated as NULL;

template<typename... Args>
vector<vector<D>> read_csv(const string path, bool header = true, Args&&...){
	cout<<endl;
	print_header("read_csv");

	print_top();

	ifstream file(path);
    string line,value;

    vector<vector<D>> data;             // Final output matrix
	vector<string> headers;				// String of header for columns
	vector<map<string,D>> replace;		// Efficient replace than using maps to identify column headers
	vector<bool> numeric;    			// determines if a certain column is float or not

	if(header == true){
		getline(file,line);
        stringstream ss(line);

		while(getline(ss, value, ',')){
			headers.push_back(value);
			replace.push_back(dummy_replace[value]);
		}
	}

		getline(file,line);
	    stringstream ss(line);
		vector<D> row;

		for(int col=0; getline(ss, value, ','); col++){
			if((header == true) && (!replace[col].empty())){
				numeric.push_back(true);
				if(replace[col].find(value) == replace[col].end())
					row.push_back(NAN);
				else
					row.push_back(replace[col][value]);
			}
			else{
				if(find(null_values.begin(), null_values.end(), value) == null_values.end()){
					try{
						row.push_back(stod(value));
						numeric.push_back(true);
					}
					catch(const exception& e){
						numeric.push_back(false);
					}
				}
				else{
					row.push_back(NAN);
					numeric.push_back(true);
				}
			}
		}

		data.push_back(row);

	if(header == false)
		replace.resize(numeric.size(),dummy_replace["cjqnorvby"]);

    while(getline(file, line)){
        stringstream ss(line);

		for(int col=0,c=0; getline(ss, value, ','); col++){
			if(numeric[col] == true){
				if(replace[col].empty()){
					if(find(null_values.begin(), null_values.end(), value) == null_values.end())
						row[c++] = stod(value);
					else
						row[c++] = NAN;
				}
				else
					row[c++] = replace[col][value];
			}
		}

		data.push_back(row);
    }

    print(line = "Included Columns respectively are:");
    line = "[";
    for(int i=0;i<headers.size();i++)
    	if(numeric[i] == 1)
			line += "'" + headers[i] + "',";
	line.pop_back();
	if(!line.empty()){
		line.pop_back();
		line.push_back(']');
	}
	print(line);
	print(line = "");
	print(line = "Discarded Columns are:");
	line = "[";
    for(int i=0;i<headers.size();i++)
    	if(numeric[i] == 0)
			line += "'" + headers[i] + "',";
	line.pop_back();
	if(!line.empty()){
		line.pop_back();
		line.push_back(']');
	}
	print(line);

	print_bottom();
	cout<<endl;

	dummy_replace.clear();
	null_values = {""};

    return data;
}

void write_csv(const string path, vector<D> data);
