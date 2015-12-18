
#include "RL.h"

float EPSILON0 = 0.2;
/**
constructor: initialize some v.r.
*/
RLearn::RLearn()
{
        qList.clear();
        _indexCur = 0;

}
void RLearn::rLearnInit(size_t numS, size_t numA, float ovsTarget)
{
    _numS = numS;
    _numA = numA;
    _ovsTarget = ovsTarget;
    _indexCur = 0;
}

void RLearn::update(float state, float ovs)
{
    _state = _nextState;
    _action = _actionSuggest;

    // find the index of the current state and action
    size_t stateIndexCur = findIndex(_states,_state);
    size_t actionIndexCur = findIndex(_actions,_action);

    //cout<<"* state: "<<_state<<" * action: "<<_action<<" * stateIndexCur: "<<stateIndexCur<<"* actionIndexCur+++++++++++++++++++++++++++++++++" <<actionIndexCur<<endl;
    _indexCur = stateIndexCur*_numA + actionIndexCur;
    cout<<"* _indexCur: "<<_indexCur<<" * action: "<<_action<<endl;

    _nextState = findDisceteValue(state, 1);
    //_action = findDisceteValue(action, 0);;
    //_qold = 0;

    // compute instant reward
    reward(ovs);

    cout<<"* nextstate: "<<_nextState<<" * Reward: "<<_reward<<endl;

    //cout<<"* Reward: "<<_reward<<endl;
}

void RLearn::reward(float ovs)
{
    /*if(fabs(ovs - _ovsTarget) < 0.0005)
    {
        //_reward = 0.0001*1.0/float(pow(0.0005,2.0));
        _reward = 0.1*1.0/0.0005;
    }
    else if(ovs>_ovsTarget) // small reward
        _reward = 0.01*1.0/fabs(ovs - _ovsTarget);
    else // big reward
    {
        _reward = 0.1*1.0/fabs(ovs - _ovsTarget);
    }*/

    //if(fabs(ovs-_ovsTarget)>0.5*_ovsTarget)
    float C1 = (ovs - _ovsTarget);
    if(ovs>_ovsTarget) // small reward
    {
       // _reward = -10*fabs(ovs - _ovsTarget);

        _reward = 0.1/float(exp(pow(C1*100.0,2.0)));
    }
    else
    {
       // _reward = -5*fabs(ovs - _ovsTarget);
        _reward = 1.0/float(exp(pow(C1*100.0,2.0)));
    }
   // _reward = 0.0001*1.0/float(pow((ovs - _ovsTarget),2.0));

   cout<<"* diff: "<<fabs(ovs - _ovsTarget)<<" * reward: "<<_reward<<endl;
}


void RLearn::qLearn()
{


    float qcur = 0.0;
    float total = _reward + GAMMA*_qmax;

    cout<<"* _indexCur: "<<_indexCur<<" * Q value: "<<qList[_indexCur].qValue<<endl;

    qcur = (1-ALPHA)*qList[_indexCur].qValue+ ALPHA*total;

    // update the entry in table
    qList[_indexCur].qValue = qcur;
    //_qold = qcur;
    //cout<<"* _indexCur: "<<_indexCur<<" * Q value: "<<qcur<<endl;
    //return qcur;
}

float RLearn::selectActionSuggest()
{
    cout<<"* compute EPSILON0! **********************"<<endl;
    computeEPSILON0();
    // find the index of next state
    size_t stateIndexNext = findIndex(_states,_nextState);
    size_t stateQStart = stateIndexNext*_numA;
    cout<<"* stateIndexNext: "<<stateIndexNext<<" stateQStart: "<<stateQStart<<endl;
    // use epsilon greedy policy
    float randx = float((rand()%100))/100.0;
    //cout<<"* randx: "<<randx<<endl;
    if(randx < EPSILON0)
    {
        // take a random action for current state
        //size_t qListSize = qList.size();
        size_t randa = rand()%_numA;
        //cout<<"* randa: "<<randa<<endl;
        QEntry qEntry = qList[stateQStart+randa];
        _actionSuggest = qEntry.action;
        _qmax = qEntry.qValue;

        //_indexCur = stateQStart+randa;

        cout<<"* randx<epsilon: "<<_actionSuggest<<endl;
    }
    else
    {
        // find the action with maximum q values
        QEntry qEntry = maxEntry(stateQStart, _nextState);
        _actionSuggest = qEntry.action;
        _qmax = qEntry.qValue;
        //cout<<"* randx>epsilon: "<<_actionSuggest<<endl;


    }
    return _actionSuggest;
}


QEntry RLearn::maxEntry(size_t beginI, float state)
{

    size_t index = beginI;
    QEntry qEntry;

    if(beginI >= 0)
    {

        float maxQvalue = qList[beginI].qValue;

        for(size_t i = beginI; i < beginI+_numA; i++)
        {
            // find all qentry with the same state
            if(qList[i].state == state)
            {
                //cout<<"* i:"<<i<<" * qList[i].qValue: "<<qList[i].qValue<<" state: "<<maxQvalue<<endl;
                if(qList[i].qValue > maxQvalue)
                {
                    maxQvalue = qList[i].qValue;
                    index = i;
                    //cout<<"max:::"<<" "<<endl;
                }
                cout<<"* i: "<<i<<" *: "<<qList[i].qValue<<"     ";
            }

        }
    }
    else
    {
        cout<<"* RLearn::maxEntry:find nothing!"<<endl;
    }
    //_indexCur = index;
    qEntry = qList[index];
    cout<<"* max index: "<<index<<endl;

    if(index == -1)
    {
        cout<<"* RLearn::maxEntry:wrong!"<<endl;
    }

    cout<<endl;
    //cout<<"* RLearn::maxEntry:_indexCur: "<<_indexCur<<endl;

    return qEntry;


}


size_t RLearn::findIndex(vector<float>& vec, float value)
{
    size_t vecSize = vec.size();
    size_t index = -1;

    for(size_t i = 0; i < vecSize; i++)
    {
        if(vec[i] == value)
        {
            index = i;
            break;
        }
        //cout<<"vec[i]"<<vec[i]<<endl;
    }

    if(index == -1)
    {
        cout<<"* RLearn::maxEntry:wrong!"<<endl;
    }

    //cout<<"* vecSize: "<<vecSize<<" * RLearn::findIndex:index+++++++++++++++++++++++++++: "<<index<<endl;
    return index;

}//end findEntry

void RLearn::initQtable(float minState, float maxState, float minAction, float maxAction
, size_t numS,  size_t numA,  float ovsTarget)
{
    //init some values first
    rLearnInit(numS,  numA,  ovsTarget);

    float granuS = (maxState- minState)/_numS;
    float granuA = (maxAction- minAction)/_numA;

    // initialize states and actions
    for(size_t i = 0; i < _numS; i++)
    {
        float stateTmp = minState + i*granuS;
        _states.push_back(stateTmp);
    }

    for(size_t i = 0; i < _numA; i++)
    {
        float actionTmp = minAction + i*granuA;
        _actions.push_back(actionTmp);
    }

    // discrete states
    vector<QEntry> qvec;

    for(size_t i = 0; i < _numS; i++)
    {
        qvec.clear();
        float stateTmp = minState + i*granuS;
        // discrete actions
        for(size_t j = 0; j < 3; j++)
        {
            float actionTmp;

            // no change
            if (j == 0)
                actionTmp = stateTmp;
            else
            {
                actionTmp = stateTmp + (-1)^j*granuA;  // indeament or decrement
            }


            // init qList
            QEntry qEntry;
            qEntry.action = actionTmp;
            if(actionTmp < minAction || actionTmp > maxAction)
            {
                qEntry.qValue = -1;
            }
            else
                qEntry.qValue = 0;

            qEntry.state = stateTmp;

            qvec.push_back(qEntry);
            //cout<<"* state: "<<stateTmp<<" action: "<<actionTmp<<endl;
        }
        qList.push_back(qvec);
    }

    _actionSuggest = minAction;
    _nextState = minState;

    //cout<<" _indexCur: "<<_indexCur<<endl;
}

/**
Given a value, find the corresponding descrete one
*/
float RLearn::findDisceteValue(float value, bool isState)
{
    // find the most close one
    float minDist = 1000000;
    vector<float> vec;
    size_t vecSize = 0;
    float valueOut = value;

    if(isState == 1)    // is a state
    {
         vec = _states;
         //vecSize = _numS;

    }
    else
    {
        vec = _actions;
        //vecSize = _numA;
    }

    vecSize = vec.size();

    for(size_t i = 0; i < vecSize; i++)
    {
        float distTmp = fabs(value-vec[i]);
        if(distTmp<minDist)
        {
            valueOut = vec[i];
            minDist = distTmp;
        }
    }
    return valueOut;
}

void RLearn::computeEPSILON0()
{
    int nonZeroSize = 0;
    for(int i = 0; i < _numA; i++)
    {
        if(qList[i].qValue != 0)
            nonZeroSize ++;
    }
    EPSILON0 = pow(0.5,nonZeroSize/2.0);
    cout<<"* nonZeroSize: "<<nonZeroSize<<" * EPSILON0: "<<EPSILON0<<endl;
    //return EPSILON0;

}


