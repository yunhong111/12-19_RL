
#ifndef RLearnH_
#define RLearn_H_

#include <stdio.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <string>
#include <cstring>
#include <sstream>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <math.h>
#include <iterator>
# include <omp.h>

using namespace std;
/* Place to put all of my definitions etc. */

// discount rate
const float GAMMA = 0.5;

// learn rate
const float ALPHA = 0.2;

// epsilon greedy
extern float EPSILON0;


struct QEntry{
    //size_t stateID;
    float state;
    float action;
    float qValue;

};

class RLearn
{
    // construction
public:
    RLearn();
    ~RLearn();

    // variables
public:

    // the old q value for the current state and action pair
    float _qold;

    // the maximum q value for all possible actions
    float _qmax;

    // the current state
    float _state;

    // the current action
    float _action;

    // the next state
    float _nextState;

    // instant reward
    float _reward;

    // the suhested action
    float _actionSuggest;

    // #states
    size_t _numS;

    // #actions
    size_t _numA;

    // ovs target
    float _ovsTarget;

    // a list storing all q values
    vector<vector<QEntry> > qList;

    // a list storing all states
    vector<float> _states;

    // a list storing all actions
    vector<float> _actions;

private:
    // the current index of entry
    size_t _indexCur;

public:
    // methods

    void rLearnInit(size_t numS, size_t numA, float ovsTarget);

    // Update q value
    void qLearn();

    float selectActionSuggest();

    void reward(float ovs);

    void update(float state,  float ovs);

    void initQtable(float minState, float maxState, float minAction, float maxAction
    , size_t numS,  size_t numA,  float ovsTarget);

    QEntry maxEntry(size_t beginI, float state);

    size_t findIndex(vector<float>& vec, float value);

    float findDisceteValue(float value, bool isState);

    void computeEPSILON0();

};

#endif
