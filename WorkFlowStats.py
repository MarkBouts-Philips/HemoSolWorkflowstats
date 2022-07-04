##
#
#
import collections
import sys
import pandas as pd
import json
from matplotlib import pyplot as plt
from collections import Counter
import seaborn as sns
import datetime
import numpy as np

# event structure
#
# id
# name
# category
# payload
##  type
##  text
##  path
##  ts :
#  startedAt :

def ParseEvents(events_log,category='Action') :
    """

    :param event:
    :return:
    """

    ### Convert events to dataframe

    _df = pd.DataFrame.from_dict(events_log)
    _df.rename(columns={'id':'event_id','name':'event_name'},inplace=True)
    ### Convert category dicts to columns
    cat_df = _df['category'].apply(pd.Series)
    cat_df.rename(columns={'id':'cat_id','name':'cat_name'},inplace=True)
    _dfe = pd.concat([_df.drop(['category'],axis=1),cat_df],axis=1)
    ### Derive the duration of the event in seconds
    _dfe['deltaTime'] = abs(pd.to_datetime(_dfe.iloc[:, 4]) - pd.to_datetime(_dfe.iloc[:, 3])).dt.total_seconds()
    ## Group start by minute
    _dfe['startblock'] = pd.to_datetime(_dfe.iloc[:, 3]).dt.hour * 24 * 60 + pd.to_datetime(_dfe.iloc[:, 3]).dt.minute * 60 + round(pd.to_datetime(_dfe.iloc[:, 3]).dt.second / 5)
    _dfe['endblock'] = pd.to_datetime(_dfe.iloc[:, 4]).dt.hour * 24 * 60 + pd.to_datetime(_dfe.iloc[:, 4]).dt.minute * 60 + round(pd.to_datetime(_dfe.iloc[:, 4]).dt.second / 5)
    _dfe['deltablock'] = _dfe['startblock'] + _dfe['deltaTime']

    # ### Every action starts with the action taker
    # mergedidx = (qq1 + qq2).dropna().apply(lambda k: k[0].union(k[1]))
    ## Define the grouping
    _dfe['group'] = 0
    _dfe['actiongroup'] = 0
    ### First group according to action taker
    selblocks = _dfe.loc[_dfe['cat_name']=='Actiontaker'].groupby('startblock').aggregate(e)['endblock'].apply(lambda r: pd.Index(r[0]))
    globalcount = 1
    ### Loop over the action takers
    for count, selblock in enumerate(selblocks) :
        print('Group {}'.format(count))
        _dfe.loc[_dfe['startblock'].between(_dfe.loc[selblock]['startblock'].iloc[0], _dfe.loc[selblock]['endblock'].iloc[0]), 'group'] = count+1
        ### Per Actiontaker determine whether there where multiple events and group accordingly
        sel_at = _dfe.loc[_dfe['startblock'].between(_dfe.loc[selblock]['startblock'].iloc[0],
                                            _dfe.loc[selblock]['endblock'].iloc[0])]
        print('Identified {} actions for this Actiontaker'.format(sum(sel_at['cat_name']==category)))
        if sum(sel_at['cat_name']==category) :
            atblocks = sel_at.loc[sel_at['cat_name'] == category]
            ## @TODO inefficient please change
            for atidx, selatblock in atblocks.iterrows() :
                #selatblock = sel_at.loc[sel_at['cat_name']=='Action'].iloc[1]
                print('Actiongroup {} {}'.format(globalcount,atidx))
                ## @TODO really find who is intersecting we are now skipping information
                _dfe.loc[_dfe['startblock'].between(selatblock['startblock'],selatblock['endblock']),'actiongroup']=globalcount
                globalcount = globalcount + 1
    return _dfe

def FormattedPrint(_dfe,actiongroupnumber) :
    """

    :param sel_event:
    :return:
    """
    if actiongroupnumber :
        sel_event = _dfe.loc[_dfe['actiongroup'] == actiongroupnumber]
        sel_initiator = (sel_event.loc[sel_event['cat_name'] == 'Actiontaker']['event_name']).str.cat(sep=" ")

        sel_action=(sel_event.loc[sel_event['cat_name']=='Action']['event_name']).str.cat(sep=" ")
        sel_person = (sel_event.loc[sel_event['cat_name'] == 'Person']['event_name']).str.cat(sep=" ")
        sel_location = (sel_event.loc[sel_event['cat_name'] == 'Location']['event_name']).str.cat(sep=" ")
        sel_time = pd.to_datetime(sel_event.loc[sel_event['cat_name']=='Action','startedAt']).dt.strftime('%H:%M:%S').str.cat()
        sel_duration = round(sel_event.loc[sel_event['cat_name']=='Action','deltaTime'],2).to_string(index=False)
        ### Initiator can do multiple action so look to the upper level
        if not sel_initiator :
            sel_initiator = _dfe.loc[(_dfe['group']==sel_event.iloc[0]['group']) & (_dfe['cat_name']=='Actiontaker'),'event_name'].str.cat(sep=' ')
        if not sel_location :
            sel_location = _dfe.loc[(_dfe['group']==sel_event.iloc[0]['group']) & (_dfe['cat_name']=='Location'),'event_name'].str.cat(sep=' ')
        print('{} : {} {} with {} at {} for {} seconds'.format(sel_time,sel_initiator,sel_action,sel_person,sel_location,sel_duration))


def ActionTakerStats(_dfe,group="Action"):
    """
    Split out over action takers

    :param _dfe:
    :param act:
    :return:
    """

    ## select the indexes based on Actiontaker
    selactsidx = _dfe.loc[_dfe['cat_name'] == 'Actiontaker'].groupby('event_name').aggregate(e)['startblock'].apply(
        lambda r: pd.Index(r[0]))
    allacts = []
    for i in range(selactsidx.shape[0]):
        acts = list(_dfe.loc[selactsidx.iloc[i], 'group'])
        #qq = _dfe.loc[(_dfe['group'].isin(acts)) & (_dfe['cat_name'] == 'Action'), ['event_name', 'deltablock']].groupby(
        #    'event_name').agg({'deltablock': ["count", "mean", "std"]})['deltablock']
        dfsubset = _dfe.loc[(_dfe['group'].isin(acts)) & (_dfe['cat_name'] == group), ['deltaTime']].agg({'deltaTime': ["count", "mean", "std"]})['deltaTime']
        #qq.reset_index(inplace=True)
        dfsubset['ACT'] = (selactsidx.index[i])
        allacts.append(dfsubset)
    allacts = pd.DataFrame(allacts)
    return (allacts)

def ActionTakerGroupedStats(_dfe,group='event_name') :
    """
    Split out the events per action taker

    :param _dfe:
    :param group:
    :return:
    """

    ## select the indexes based on Actiontaker
    selactsidx = _dfe.loc[_dfe['cat_name'] == 'Actiontaker'].groupby('event_name').aggregate(e)['startblock'].apply(
        lambda r: pd.Index(r[0]))
    allacts = []
    for i in range(selactsidx.shape[0]) :
        acts = list(_dfe.loc[selactsidx.iloc[i],'group'])
        subsetdf=_dfe.loc[(_dfe['group'].isin(acts))&(_dfe['cat_name']==group),['event_name','deltaTime']].groupby('event_name').agg({'deltaTime' : ["count","mean","std"]})['deltaTime']
        subsetdf.reset_index(inplace=True)
        subsetdf['ACT'] = (selactsidx.index[i])
        allacts.append(subsetdf)
    allacts = pd.concat(allacts,axis=0)
    return(allacts)

def e(inp):
    return [inp.index]

# def PlotActionTakerOverview(x,val='mean') :
#     """
#
#     :param x:
#     :param val:
#     :return:
#     """
#     plt.figure(figsize=(10, 10))
#     # ax = sns.catplot(data=x, x='ACT', y=val,yerr=x['std'].values, kind="bar", hue='ACT')
#      ax = sns.barplot(data=x, x='ACT', y=val,yerr=x['std'].values*1, hue='ACT')
#     # ax = sns.barplot(x['ACT'].values, x[val].values, yerr=x['std'].values)
#     plt.bar(x['ACT'].values,x[val].values,yerr=x['std'].values*1)
#     #ax = sns.barplot(data=x, x='ACT', y=val, hue='ACT')
#     # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#     plt.xticks(rotation=90,ha='center')
#     plt.tight_layout()
#     plt.legend()
#     plt.show()

def GroupedBarPlot(df,cat='event_name',subcat='ACT',val='mean',err='std',title="Actiontakers actions",outfileprefix=None) :
    """

    :param df:
    :return:
    """

    def fix_df(_ddf,cats):
        missing_data = set(cats).difference(set(_ddf[cat].unique()))
        if len(missing_data) :
            offset = _ddf.shape[0]
            _ddf = pd.concat([_ddf]*(len(missing_data)+1),ignore_index=True)
            for i,d in enumerate(missing_data):
                _ddf.iloc[offset+i,[0]] = d
                _ddf.iloc[offset+i,[1,2,3]] = 0
        return _ddf

    cats = df[cat].unique()
    subx = df[subcat].unique()
    x = np.arange(len(cats))
    offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
    width = np.diff(offsets).mean()

    plt.figure()
    for i, gr in enumerate(subx):
        ddfg = df[df[subcat] == gr]
        dfg=fix_df(ddfg,cats)
        plt.bar(x + offsets[i], dfg[val].values,width=width,
                label="{} {}".format(subcat, gr), yerr=dfg[err].values)
    plt.xlabel(cat)
    plt.ylabel("Average time (s)")
    plt.xticks(x, cats,rotation=90,ha='center')
    plt.legend()
    plt.title(label=title)
    plt.tight_layout()
    #
    if outfileprefix :
        plt.savefig(''.join(['C:/Users/320046082/tmp/',outfileprefix, 'AverageTimePerEvent.png']), dpi=300, facecolor=None)
    else :
        plt.show()

def PlotEventOverview(x,val='mean') :
    """

    :param x:
    :return:
    """

    fig = plt.figure(figsize=(10, 10))

    #ax = sns.catplot(data=x, x='event_name', y=val,yerr='std', kind="bar", hue='ACT')
    ax = sns.catplot(data=_dfe, x='event_name', y='deltaTime', kind="bar", hue='ACT',legend_out=True)
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
   # ax.legend(loc="lower left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    #ax.legend(loc="lower left", bbox_to_anchor=(0.1, -0.4))

    plt.show()

def PlotEventHistogram(events_log) :
    """
    :param events : list of all
    :return:
    """
    actions = []
    ts = []
    for event in events_log :
        actions.append(event['name'])
        ts.append(pd.to_datetime(event['startedAt']))

    counts = Counter(actions)
    x = pd.DataFrame.from_dict(counts,orient='index').reset_index()
    x = x.rename(columns={"index":"action",0:"count"})
    plt.figure(figsize=(12,10))
    plt.xticks(rotation=90,ha='center')
    #plt.bar(counts.keys(),counts.values())
    ax = sns.barplot(data=x, x="action", y="count")
    plt.tight_layout()
    plt.savefig(''.join(['C:/Users/320046082/tmp/','Test2.png']),dpi=300,facecolor=None)

file="C:/Users/320046082/OneDrive - Philips/Documents/projects/ICU/HemoSolutions/data/WorkflowLogger/Handover2/Handover.json"

with open(file,'r') as wffile :
    wfdata = wffile.read()

wfobj = json.loads(wfdata)
events_log = wfobj['events']
print('Identified {} events'.format(str(len(events_log))))

_dfe = ParseEvents(events_log)
### Are their multiple actions conducted by this Actiontaker during this timeperiod?
for grp in _dfe['actiongroup'].unique():
    FormattedPrint(_dfe, grp)


## Select the actions and summarize
_dfe.loc[_dfe['cat_name'] == 'Action'].groupby('event_name').agg({ 'deltaTime' :['count','mean','std']})
    ### With whom is the action undertaken?
_dfe.loc[_dfe['cat_name'] == 'Person'].groupby('event_name').agg({ 'deltaTime' :['count','mean','std']})
    ### Where were the actions undertaken?
_dfe.loc[_dfe['cat_name'] == 'Location'].groupby('event_name').agg({'deltaTime': ['count', 'mean', 'std']})


act_data = ActionTakerGroupedStats(_dfe, 'Action')
act_data = act_data.fillna(0)
GroupedBarPlot(df=act_data, cat='event_name', subcat='ACT', val='mean', err='std',title="Actions taken by actiontaker (ACT)",outfileprefix="ActionGrouping")
act_data = ActionTakerGroupedStats(_dfe, 'Person')
act_data = act_data.fillna(0)
GroupedBarPlot(df=act_data, cat='event_name', subcat='ACT', val='mean', err='std',title="Actiontaker (ACT) communicates with",outfileprefix="PersonGrouping")
act_data = ActionTakerGroupedStats(_dfe, 'Object')
act_data = act_data.fillna(0)
GroupedBarPlot(df=act_data, cat='event_name', subcat='ACT', val='mean', err='std',title="Actiontaker (ACT) uses",outfileprefix="ObjectGrouping")

### Some other stuff
PlotEventHistogram(events_log)
act_data = ActionTakerStats(_dfe)
act_data = act_data.fillna(0)
PlotActionTakerOverview(act_data, 'mean')


