#include <LightGBM/c_api.h>
#include <csv2/writer.hpp>
#include <csv2/reader.hpp>
#include <LightGBM/boosting.h>

#include <Python.h>

#include <map>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <vector>
#include <string>

void generate_samples() {
    // ready for writer
    std::ofstream stream("lgb.csv");
    csv2::Writer<csv2::delimiter<','>> writer(stream);

    // init hotness values
    std::map<int, double> hotness_map;
    double base_hotness = 0.1;
    for (int i=0; i<200; i++) {
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) + base_hotness;
        hotness_map[i] = r;
    }

    // init header vector
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> header;
    header.emplace_back("Level");
    for (int i=0; i<20; i++) {
        header.emplace_back("Range_" + std::to_string(i));
        header.emplace_back("Hotness_" + std::to_string(i));
    }
    header.emplace_back("Target");
    rows.emplace_back(header);

    // ready for shuffling
    std::vector<int> ids;
    for(int i=0; i<200; i++) {
        ids.emplace_back(i);
    }

    // generate values
    for (int i=0; i<1000; i++) {
        // std::vector<double> value;
        std::vector<std::string> values;
        int level = i / 200;
        int target = 5 - level;
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        if (r > 0.10 * level) {
            target -= 1;
        }

        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(ids.begin(), ids.end(), std::default_random_engine(seed));
        values.emplace_back(std::to_string(level));
        for (int i=0; i<20; i++) {
            values.emplace_back(std::to_string(ids[i]));
            values.emplace_back(std::to_string(int(1e8 * hotness_map[ids[i]])));
        }
        values.emplace_back(std::to_string(target));

        rows.emplace_back(values);
    }

    writer.write_rows(rows);
    stream.close();
}

void read_samples() {
    csv2::Reader<csv2::delimiter<','>, 
                csv2::quote_character<'"'>, 
                csv2::first_row_is_header<true>,
                csv2::trim_policy::trim_whitespace> csv;
               
    if (csv.mmap("lgb.csv")) {
        const auto header = csv.header();
        for (const auto row: csv) {
            for (const auto cell: row) {
                std::string value;
                cell.read_value(value);
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
    }

}

void train() {
    PyObject* pModule = PyImport_ImportModule("lgb");
	if( pModule == nullptr ){
		std::cout <<"module not found" << std::endl;
		exit(EXIT_FAILURE);
	}

    PyObject* pFunc = PyObject_GetAttrString(pModule, "train");
	if( !pFunc || !PyCallable_Check(pFunc)){
		std::cout <<"not found function" << std::endl;
		exit(EXIT_FAILURE);
	}

    PyObject* pArg = PyTuple_New(2);
    PyTuple_SetItem(pArg, 0, Py_BuildValue("s", "lgb.csv")); 
    PyTuple_SetItem(pArg, 1, Py_BuildValue("s", "lgb.txt")); 

    PyObject_CallObject(pFunc, pArg);

    Py_DECREF(pModule);
    Py_DECREF(pFunc);
    Py_DECREF(pArg);
}

/*
uint16_t predict_one() {
    std::vector<uint32_t> sample = {
        0,77,83853435,86,32896816,164,109999358,88,
        45036017,191,97761380,192,84780931,40,62674498,
        71,13928034,187,85729384,85,43033713,95,
        102396976,93,95867633,185,19964006,154,62021011,21,
        34288677,161,85558086,181,65248507,162,15193881,
        136,22547489,99,101097202
    };

    PyObject* pModule = PyImport_ImportModule("lgb");
	if( pModule == nullptr ){
		std::cout <<"module not found" << std::endl;
		exit(EXIT_FAILURE);
	}

    PyObject* pFunc = PyObject_GetAttrString(pModule, "predict");
	if( !pFunc || !PyCallable_Check(pFunc)){
		std::cout <<"not found function" << std::endl;
		exit(EXIT_FAILURE);
	}

    PyObject* pArg = PyTuple_New(2);
    PyTuple_SetItem(pArg, 0, Py_BuildValue("s", "lgb.txt")); 

    PyObject* pData = PyList_New(0);
    for (int feature : sample) {
        PyList_Append(pData, Py_BuildValue("i", feature));
    }
    PyTuple_SetItem(pArg, 1, pData); 

    PyObject* pReturn = PyObject_CallObject(pFunc, pArg);

    int nResult;
    PyArg_Parse(pReturn, "i", &nResult);

    // std::cout << "return result is " << nResult << std::endl;
    return nResult;
}
*/

void predict(std::vector<uint16_t>& results) {
    results.clear();
    std::vector<std::vector<uint32_t>> samples = {
        {
            0,77,83853435,86,32896816,164,109999358,88,
            45036017,191,97761380,192,84780931,40,62674498,
            71,13928034,187,85729384,85,43033713,95,
            102396976,93,95867633,185,19964006,154,62021011,21,
            34288677,161,85558086,181,65248507,162,15193881,
            136,22547489,99,101097202
        },
        {
            2,113,32610663,147,83265441,100,58249068,136,22547489,
            166,98995566,141,105010402,99,101097202,146,89779806,
            102,105025231,21,34288677,49,104932701,126,78444504,25,
            50094437,48,16975528,1,49438291,191,97761380,31,
            93911224,107,53195345,129,46866354,111,40745785
        },
        {
            4,125,103500401,33,39603161,64,36666575,75,82095235,182,
            67943000,42,50022864,96,49843665,148,75656366,18,24160255,
            57,12002304,110,88600212,185,19964006,8,37777471,16,73571175,
            26,22979043,153,23490241,104,24766001,100,58249068,
            137,89347040,69,76772379
        }
    };

    PyObject* pModule = PyImport_ImportModule("lgb");
	if( pModule == nullptr ){
		std::cout <<"module not found" << std::endl;
		exit(EXIT_FAILURE);
	}

    PyObject* pFunc = PyObject_GetAttrString(pModule, "predict");
	if( !pFunc || !PyCallable_Check(pFunc)){
		std::cout <<"not found function" << std::endl;
		exit(EXIT_FAILURE);
	}

    PyObject* pArg = PyTuple_New(2);
    PyTuple_SetItem(pArg, 0, Py_BuildValue("s", "lgb.txt")); 

    PyObject* pDatas = PyList_New(0);
    PyObject* pData = nullptr;
    size_t cnt = 0;
    for (std::vector<uint32_t>& sample : samples) {
        pData = PyList_New(0);
        for (uint32_t& feature : sample) {
            PyList_Append(pData, Py_BuildValue("i", feature));
        }
        PyList_Append(pDatas, pData);
        cnt += 1;
    }
    
    PyTuple_SetItem(pArg, 1, pDatas); 

    PyObject* pReturn = PyObject_CallObject(pFunc, pArg); // should return list

    for (size_t i = 0; i < cnt; i ++) {
        int nResult = 0;
        PyArg_Parse(PyList_GetItem(pReturn, i), "i", &nResult);
        results.emplace_back(nResult);
    }
    
    Py_DECREF(pModule);
    Py_DECREF(pFunc);
    Py_DECREF(pArg);
    Py_DECREF(pDatas);
    Py_DECREF(pReturn);
}

int main() {
    Py_Initialize();
	if(!Py_IsInitialized()){
		std::cout << "python init fail" << std::endl;
		exit(EXIT_FAILURE);
	}

    PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('.')");

    generate_samples(); 
    train();

    std::vector<uint16_t> results;
    predict(results);
    for (uint16_t result : results) {
        std::cout << result << " " << std::endl;
    }
    std::cout << std::endl;
    //read_samples();

    Py_Finalize();

    return EXIT_SUCCESS;
}