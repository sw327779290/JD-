import os
os.chdir(r'C:\Users\PC\Desktop\competition\JD\data_ori')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


user_file = "JData_User.csv"
product_file = "JData_Product.csv"
action_1_file = "JData_Action_201602.csv"
action_2_file = "JData_Action_201603.csv"
action_3_file = "JData_Action_201604.csv"
comment_file = "JData_Comment.csv"


def get_basic_user_feat():
    if os.path.exists('../cache/cache_' + user_file):
        return pd.read_csv('../cache/cache_' + user_file)
    else:
        print('------>Gernearting  cache_%s.csv' %  user_file)
        user = pd.read_csv(user_file, encoding='gbk')

        age_map = {
            u'-1': 0,
            u'15岁以下': 1,
            u'16-25岁': 2,
            u'26-35岁': 3,
            u'36-45岁': 4,
            u'46-55岁': 5,
            u'56岁以上': 6,
        }
        user['age'] = user['age'].map(age_map).fillna(-1)
        #没有注册时间的使用频数2015-11-11填充
        #对注册时间分别统计年份，月份，天数，星期
        user['user_reg_tm'] = user['user_reg_tm'].fillna('2015-11-11').astype(str)
        user['user_regist_year'] = user['user_reg_tm'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').year)
        user['user_regist_month'] = user['user_reg_tm'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').month)
        #user['user_regist_day'] = user['user_reg_tm'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').day)
        user['user_regist_weekday'] = user['user_reg_tm'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').weekday())
        clean_user = [user['user_id'],]
        for column in ['age', 'sex', 'user_lv_cd','user_regist_year','user_regist_month','user_regist_weekday']:
            clean_user.append(
                pd.get_dummies(user[column], prefix=column))
        clean_user = pd.concat(clean_user, axis=1)
        clean_user.to_csv('../cache/cache_' + user_file, index=False)

        return clean_user

def dim_expan_with_cache(file, ori_cols, cols):
    if os.path.exists('../cache/cache_' + file):
        return pd.read_csv('../cache/cache_' + file)
    else:
        print('------>Gernearting  cache_%s.csv' %  file)
        df = pd.read_csv(file)
        expan_df = [df[ori_cols], ]
        for column in cols:
            expan_df.append(
                pd.get_dummies(df[column], prefix=column))
        expan_df = pd.concat(expan_df, axis=1)
        expan_df.to_csv('../cache/cache_' + file, index=False)
        return expan_df

def get_basic_product_feat():
    return dim_expan_with_cache(
        product_file,
        #['sku_id','cate','brand'],
        #['a1','a2','a3']
        ['sku_id', 'cate'],
        ['a1', 'a2', 'a3','brand']
    )


def _get_actions():
    keep_cols = ['user_id', 'sku_id', 'time', 'type', 'cate', 'brand'] # model_id
    actions = []
    for f in [action_1_file, action_2_file, action_3_file]:
        action = pd.read_csv(f, parse_dates=['time'], usecols=keep_cols)
        action['user_id'] = pd.to_numeric(action['user_id'], downcast="float")
        action['sku_id'] = pd.to_numeric(action['sku_id'], downcast="signed")
        action['type'] = pd.to_numeric(action['type'], downcast="unsigned")
        action['cate'] = pd.to_numeric(action['sku_id'], downcast="unsigned")
        action['brand'] = pd.to_numeric(action['type'], downcast="unsigned")
        #action = action.drop_duplicates()
        actions.append(action)
    # 垂直堆叠 ; 改变数据类型后占用由6574M降到1447M
    return pd.concat(actions)


_actions = _get_actions()
 #_actions = None

def get_actions(start_date, end_date):
    # 没有必要来存储浪费磁盘时间
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    actions = _actions
    return actions[(actions.time >= start_date) & (actions.time < end_date)]
def conver_time(time):
    if time in range(0,8):
        return 'morining'
    if time in range(8,16):
        return 'noon'
    if time in range(16,24):
        return 'evening'

def conver_time_1(time):
    if 0<= time & time <2:
        return 'one'
    if 2<= time & time <4:
        return 'two'
    if 4<= time & time <6:
        return 'three'
    if 6<= time & time <8:
        return 'four'
    if 8<= time & time <10:
        return 'five'
    if 10<= time & time <12:
        return 'six'
    if 12<= time & time <14:
        return 'seven'
    if 14<= time & time <16:
        return 'eight'
    if 16<= time & time <18:
        return 'nine'
    if 18<= time & time <20:
        return 'ten'
    if 20<= time & time <22:
        return 'eleven'
    if 22<= time & time <24:
        return 'twelve'
def get_action_feat(start_date, end_date):
    file = '../cache/cache_action_accumulate_from_%s_to_%s.csv' % (start_date, end_date)
    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        print('------>Gernearting  cache_action_accumulate_from_%s_to_%s.csv' % (start_date, end_date))
        actions = get_actions(start_date, end_date)
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        day = (start_date - end_date).days
#---------------------------------------------------------------------------------
#将用户行为分为3个时间段
        actions['action_hour'] = actions['time'].apply(lambda x: x.hour)
        #actions['type'] = actions['type'].apply(lambda x: str(x))+ actions['action_hour']
        #del actions['action_hour']
        #actions['action_hour'] = actions['time'].map(lambda x: datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S').timetuple()[3])
        actions['action_hour'] = actions['action_hour'].map(conver_time).fillna('None')
        actions['new_type'] = actions['type'].map(lambda x: str(x))
        actions['type'] = actions['new_type'] + actions['action_hour']
        del actions['action_hour']
        del actions['new_type']
#-----------------------------------------------------------------------------------------
        # 1.浏览（指浏览商品详情页）2.加入购物车3.购物车删除4.下单5.关注6.点击
        # 将行为类型展宽为多维向量
        #actions['hour'] = actions.
        types = pd.get_dummies(actions['type'], prefix='%dth-action' % day)
        actions = pd.concat(
            [actions[['user_id', 'sku_id']], types],
            axis=1)
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        # 若as_index = True则user_id作为index
        # 汇总同一用户对同一商品的某类型行为，即同类行为叠加(在某段时间内)
        actions.to_csv(file, index=False)
        return actions

# -----------------------------------------------------------
comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07", "2016-03-14",
                "2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]

def get_comments_product_feat(start_date, end_date):
    # http://www.datafountain.cn/#/competitions/247/data-intro
    file = '../cache/cache_comments_accumulate_to_%s.csv' % end_date
    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        print('------>Gernearting cache_comments_accumulate_to_%s.csv' % end_date)
        comments = pd.read_csv(comment_file)
        comment_date_end = end_date
        # date_begin的从列表中选出，为表中刚好比date_end小的
        comment_date_begin = ''
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        # 将评论条数拓展为多维向量, comments.dt为日期
        comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
        comment_num = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, comment_num], axis=1)
        #del comments['dt']
        #del comments['comment_num']
        comments = comments[['sku_id',
                             'has_bad_comment',
                             'bad_comment_rate',
                             'comment_num_1',
                             'comment_num_2',
                             'comment_num_3',
                             'comment_num_4']]
        comments.to_csv(file, index=False)
        return comments

# --------------------------拓展用户信息---------------------------------
def get_adv_user_feat(start_date, end_date):
    file = 'cache_expand_' + end_date + user_file
    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        print('------>Gernearting  cache_expand_ %s+ %s.csv' % (end_date, user_file))
        user = get_basic_user_feat()
        action = get_actions(start_date, end_date)
        # 特征：用户的操作之间的平均时间间隔
        action.sort_values(['user_id', 'time'], inplace=True)
        diff = action.groupby(['user_id'])['time'].diff()
        action['deltatime'] = diff
        # 排除间隔大于1天的，后面可增加特征：用户是否有超过一天间隔的动作  (是否排除下单动作？)
        action2 = action[
            action['deltatime'].notnull() # 可以省略
            & (action['deltatime'] < pd.Timedelta('1 days'))
            & (action['deltatime'] > pd.Timedelta('0'))
        ]
        deltatime = action2.groupby(['user_id']).apply(
            lambda x: x['deltatime'].mean().seconds
        ).reset_index(name='mean_delta_time')
        # deltatime['mean_delta_time'] = deltatime['mean_delta_time'].apply(lambda x: x.seconds)

        s0 = pd.Timedelta('10 minutes').seconds; s1 = pd.Timedelta('30 minutes').seconds
        s2 = pd.Timedelta('1 hours').seconds; s3 = pd.Timedelta('3 hours').seconds
        def map_delta_time(x):
            if x == -1:
                return 9 # 手动处理nan
            if x <= s0:
                return 1
            if s0 < x <= s1:
                return 2
            if s1 < x <= s2:
                return 3
            if s2 < x <= s3:
                return 4
            else:
                return 5

        after_merge = pd.merge(user, deltatime, how='left', on='user_id')
        mean_delta_time = after_merge.mean_delta_time.fillna(-1).map(map_delta_time)
        new_user = pd.concat([
            after_merge,
            pd.get_dummies(mean_delta_time, prefix='mean_delta_time_every_action')
        ], axis=1)
        
        # 特征：用户的最后一次行为类型
        last_type = action.groupby(['user_id']).apply(   # , as_index=False
            lambda x: x.loc[x['time'].idxmax(), 'type']
        ).reset_index(name='last_type')
        after_merge = pd.merge(new_user, last_type, how='left', on='user_id')
        new_user = pd.concat([
            after_merge,
            pd.get_dummies(after_merge['last_type'], prefix='last_type', dummy_na=True) # 自动处理nan
        ], axis=1)
        del new_user['last_type']
        
        # 特征：用户的最后一次下单时间 （距离预测日期的时间）
        last_buy_time = action.groupby(['user_id']).apply(
            lambda x: x[x['type'] == 4]['time'].max() if (x['type'] == 4).sum() > 0 else None # pd.np.NAN
        ).reset_index(name='last_buy_time')
        
        end_date_ = datetime.strptime(end_date, '%Y-%m-%d')
        last_buy_time['last_buy_time'] = end_date_ - last_buy_time['last_buy_time']
        
        s0 = pd.Timedelta('1 hours'); s1 = pd.Timedelta('8 hours')
        s2 = pd.Timedelta('1 days'); s3 = pd.Timedelta('2 days')
        s4 = pd.Timedelta('3 days'); s5 = pd.Timedelta('5 days')
        s6 = pd.Timedelta('12 days')
        def map_delta_time2(x):
            if x <= s0:
                return 1
            if s0 < x <= s1:
                return 2
            if s1 < x <= s2:
                return 3
            if s2 < x <= s3:
                return 4
            if s3 < x <= s4:
                return 5
            if s4 < x <= s5:
                return 6
            if s5 < x <= s6:
                return 7
            else:
                return 8
        
        last_buy_time['last_buy_time'] = last_buy_time['last_buy_time'].map(
            map_delta_time2, na_action='ignore')
        
        after_merge = pd.merge(new_user, last_buy_time, how='left', on='user_id')
        new_user = pd.concat([
            after_merge,
            pd.get_dummies(after_merge['last_buy_time'], prefix='last_buy_time', dummy_na=True) # 自动处理nan
        ], axis=1)
        del new_user['last_buy_time']
        
        # 特征：用户下单的时间间隔
        # 先找到两次以上下单的用户
        two_more_buy = action.groupby(['user_id'], as_index=False).filter(
            lambda x: (x['type'] == 4).sum() > 1
        )[['user_id', 'type', 'time']]
        two_more_buy = two_more_buy[two_more_buy['type'] == 4]
        two_more_buy.sort_values(['user_id', 'time'], inplace=True)
        diff = two_more_buy.groupby(['user_id'])['time'].diff()
        two_more_buy['deltatime'] = diff
        two_more_buy = two_more_buy[two_more_buy['deltatime'].notnull()]
        deltatime = two_more_buy.groupby(['user_id']).apply(
            lambda x: x['deltatime'].mean()
        ).reset_index(name='mean_delta_time_two_more_buy')
        
        s0 = pd.Timedelta('10 hours')
        s1 = pd.Timedelta('1 days'); s2 = pd.Timedelta('2 days')
        s3 = pd.Timedelta('3 days'); s4 = pd.Timedelta('5 days')
        s5 = pd.Timedelta('10 days')
        def map_delta_time3(x):
            if x <= s0:
                return 1
            if s0 < x <= s1:
                return 2
            if s1 < x <= s2:
                return 3
            if s2 < x <= s3:
                return 4
            if s3 < x <= s4:
                return 5
            if s4 < x <= s5:
                return 6
            else:
                return 7
        deltatime['mean_delta_time_two_more_buy'] = deltatime['mean_delta_time_two_more_buy'].map(
            map_delta_time3, na_action='ignore')
        after_merge = pd.merge(new_user, deltatime, how='left', on='user_id')
        new_user = pd.concat([
            after_merge,
            pd.get_dummies(after_merge['mean_delta_time_two_more_buy'], prefix='mean_delta_time_two_more_buy', dummy_na=True) # 自动处理nan
        ], axis=1)
        
        new_user.to_csv(file, index=False)
        return new_user
        


# --------------------------拓展商品信息---------------------------------
def get_adv_product_feat(start_date, end_date):
    file = 'cache_expand_' + end_date + product_file
    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        print('------>Gernearting  cache_expand_ %s+ %s.csv' % (end_date, product_file))
        product = get_basic_product_feat()
        action = get_actions(start_date, end_date)
        # 特征：同一用户连续购买两次以上的商品, 可以深入挖掘，暂时不管
        always_buy = action[['user_id', 'sku_id', 'type']].groupby(
            ['user_id', 'sku_id'], as_index=False
        ).apply(
            lambda x: (x['type'] == 4).sum()
        ).reset_index(name='two_more_buy_product')
        always_buy2 = always_buy.groupby(['sku_id']).apply(
            lambda x: x['two_more_buy_product'].sum()
        ).reset_index(name='total_times_buy_more')
        new_product = pd.merge(product, always_buy2, how='left', on='sku_id')
        
        
        # 特征：商品头5天内平均下单时间间隔
        two_more_buy = action.groupby(['sku_id'], as_index=False).filter(
            lambda x: (x['type'] == 4).sum() > 1
        )[['sku_id', 'type', 'time']]
        two_more_buy = two_more_buy[two_more_buy['type'] == 4]
        two_more_buy.sort_values(['sku_id', 'time'], inplace=True)
        diff = two_more_buy.groupby(['sku_id'])['time'].diff()
        two_more_buy['deltatime'] = diff
        two_more_buy = two_more_buy[two_more_buy['deltatime'].notnull()]
        deltatime = two_more_buy.groupby(['sku_id']).apply(
            lambda x: x['deltatime'].mean()
        ).reset_index(name='mean_delta_time_two_more_buy_product')
        
        s0 = pd.Timedelta('1 hours'); s1 = pd.Timedelta('5 hours')
        s2 = pd.Timedelta('10 hours'); s3 = pd.Timedelta('1 days')
        s4 = pd.Timedelta('1.5 days'); s5 = pd.Timedelta('3 days')
        def map_delta_time3(x):
            if x <= s0:
                return 1
            if s0 < x <= s1:
                return 2
            if s1 < x <= s2:
                return 3
            if s2 < x <= s3:
                return 4
            if s3 < x <= s4:
                return 5
            if s4 < x <= s5:
                return 6
            else:
                return 7
        deltatime['mean_delta_time_two_more_buy_product'] = deltatime['mean_delta_time_two_more_buy_product'].map(
            map_delta_time3, na_action='ignore')
        after_merge = pd.merge(new_product, deltatime, how='left', on='sku_id')
        new_product = pd.concat([
            after_merge,
            pd.get_dummies(after_merge['mean_delta_time_two_more_buy_product'], prefix='mean_delta_time_two_more_buy_product', dummy_na=True) # 自动处理nan
        ], axis=1)
        
        new_product.to_csv(file, index=False)
        return new_product
  

'''def get_accumulate_user_feat(start_date, end_date):

    #特定用户行为特征分析

    feature = ['user_id', 'user_action_1_ratio',
               'user_action_2_ratio', 'user_action_3_ratio',
               'user_action_5_ratio', 'user_action_6_ratio']
    file = '../cache/cache_user_feat_accumulate_from_%s_to_%s.csv' % (start_date, end_date)
    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        print('------>Gernearting  cache_user_feat_accumulate_from_%s_to_%s.csv' % (start_date, end_date))
        actions = get_actions(start_date, end_date)
        types = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions['user_id'], types], axis=1)
        # 同一用户某段时间内的所有动作累计，action4为下单
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['user_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['user_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['user_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['user_action_6_ratio'] = actions['action_4'] / actions['action_6']
        actions = actions[feature]
        actions.to_csv(file, index=False)
        return actions


def get_accumulate_product_feat(start_date, end_date):

    #特定商品特征分析

    feature = ['sku_id', 'product_action_1_ratio', 'product_action_2_ratio', 'product_action_3_ratio', 'product_action_5_ratio', 'product_action_6_ratio']
    file = '../cache/cache_product_feat_accumulate_from_%s_to_%s.csv' % (start_date, end_date)
    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        print('------>Gernearting  cache_product_feat_accumulate_from_%s_to_%s.csv' % (start_date, end_date))
        actions = get_actions(start_date, end_date)
        types = pd.get_dummies(actions['type'], prefix='action')

        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_action_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['product_action_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['product_action_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['product_action_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['product_action_6_ratio'] = actions['action_4'] / actions['action_6']

        actions = actions[feature]
        actions.to_csv(file, index=False)
        return actions'''
#-----------------------------------------------------------------------------------------------------------
def one_month_user_action_mean_var(start_date, end_date):
     file = '../cache/one_month_user_action_mean_var_from%s_to_%s.csv' % (start_date, end_date)
     if os.path.exists(file):
        return pd.read_csv(file)
     else:
        print('------>Gernearting  one_month_user_action_mean_var_from%s_to_%s.csv' % (start_date, end_date))
        actions = get_actions(start_date, end_date)
        actions['day'] = actions['time'].apply(lambda x: (x- datetime.strptime(start_date, '%Y-%m-%d')).days)
        types = pd.get_dummies(actions['type'], prefix='user_action')
        actions = pd.concat([actions[['user_id','day']], types], axis=1)
       #--------------------构建x^2的期望-------------------------------------------------------
        var = actions.groupby(['user_id','day']).sum()
        var = var.apply(lambda x: x**2)
        var = var.reset_index('user_id')
        delta = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
        var = (var.groupby(['user_id']).sum()/delta).reset_index('user_id')
        var.columns = ['user_id', 'user_action_1_var','user_action_2_var','user_action_3_var','user_action_4_var','user_action_5_var','user_action_6_var']
        #------------------构建x的期望------------------------------------------------------------
        del actions['day']
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_mean'] = actions['user_action_1'] /delta
        actions['user_action_2_mean'] = actions['user_action_2']/delta
        actions['user_action_3_mean'] = actions['user_action_3']/delta
        actions['user_action_4_mean'] = actions['user_action_4']/delta
        actions['user_action_5_mean'] = actions['user_action_5']/delta
        actions['user_action_6_mean'] = actions['user_action_6']/delta
        actions = actions.reset_index('user_id')
        #-------------构建方差--------------------------------------------------------------------
        actions = pd.merge(actions,var,how='left',on='user_id')
        actions['user_action_1_var'] =  actions['user_action_1_var'] - actions['user_action_1_mean']**2
        actions['user_action_2_var'] =  actions['user_action_2_var'] - actions['user_action_2_mean']**2
        actions['user_action_3_var'] =  actions['user_action_3_var'] - actions['user_action_3_mean']**2
        actions['user_action_4_var'] =  actions['user_action_4_var'] - actions['user_action_4_mean']**2
        actions['user_action_5_var'] =  actions['user_action_5_var'] - actions['user_action_5_mean']**2
        actions['user_action_6_var'] =  actions['user_action_6_var'] - actions['user_action_6_mean']**2
        #-----------构建比率-----------------------------------------------------------------------
        actions['user_action_1_ratio'] = actions['user_action_4'] / actions['user_action_1']
        actions['user_action_2_ratio'] = actions['user_action_4'] / actions['user_action_2']
        actions['user_action_3_ratio'] = actions['user_action_4'] / actions['user_action_3']
        actions['user_action_5_ratio'] = actions['user_action_4'] / actions['user_action_5']
        actions['user_action_6_ratio'] = actions['user_action_4'] / actions['user_action_6']
        actions.to_csv(file, index= False)
        return actions
#--------------------------------------------------------------------------------------------------------
def one_month_sku_action_mean_var(start_date, end_date):
     file = '../cache/one_month_sku_action_mean_var_from%s_to_%s.csv' % (start_date, end_date)
     if os.path.exists(file):
        return pd.read_csv(file)
     else:
        print('------>Gernearting  one_month_sku_action_mean_var_from%s_to_%s.csv' % (start_date, end_date))
        actions = get_actions(start_date, end_date)
        actions['day'] = actions['time'].apply(lambda x: (x- datetime.strptime(start_date, '%Y-%m-%d')).days)
        types = pd.get_dummies(actions['type'], prefix='sku_action')
        actions = pd.concat([actions[['sku_id','day']], types], axis=1)
       #--------------------构建x^2的期望-------------------------------------------------------
        var = actions.groupby(['sku_id','day']).sum()
        var = var.apply(lambda x: x**2)
        var = var.reset_index('sku_id')
        delta = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
        var = (var.groupby(['sku_id']).sum()/delta).reset_index('sku_id')
        var.columns = ['sku_id', 'sku_action_1_var','sku_action_2_var','sku_action_3_var','sku_action_4_var','sku_action_5_var','sku_action_6_var']
        #------------------构建x的期望------------------------------------------------------------
        del actions['day']
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['sku_action_1_mean'] = actions['sku_action_1'] /delta
        actions['sku_action_2_mean'] = actions['sku_action_2']/delta
        actions['sku_action_3_mean'] = actions['sku_action_3']/delta
        actions['sku_action_4_mean'] = actions['sku_action_4']/delta
        actions['sku_action_5_mean'] = actions['sku_action_5']/delta
        actions['sku_action_6_mean'] = actions['sku_action_6']/delta
        actions = actions.reset_index('sku_id')
        #-------------构建方差--------------------------------------------------------------------
        actions = pd.merge(actions,var,how='left',on='sku_id')
        actions['sku_action_1_var'] =  actions['sku_action_1_var'] - actions['sku_action_1_mean']**2
        actions['sku_action_2_var'] =  actions['sku_action_2_var'] - actions['sku_action_2_mean']**2
        actions['sku_action_3_var'] =  actions['sku_action_3_var'] - actions['sku_action_3_mean']**2
        actions['sku_action_4_var'] =  actions['sku_action_4_var'] - actions['sku_action_4_mean']**2
        actions['sku_action_5_var'] =  actions['sku_action_5_var'] - actions['sku_action_5_mean']**2
        actions['sku_action_6_var'] =  actions['sku_action_6_var'] - actions['sku_action_6_mean']**2
        #-----------构建比率-----------------------------------------------------------------------
        actions['sku_action_1_ratio'] = actions['sku_action_4'] / actions['sku_action_1']
        actions['sku_action_2_ratio'] = actions['sku_action_4'] / actions['sku_action_2']
        actions['sku_action_3_ratio'] = actions['sku_action_4'] / actions['sku_action_3']
        actions['sku_action_5_ratio'] = actions['sku_action_4'] / actions['sku_action_5']
        actions['sku_action_6_ratio'] = actions['sku_action_4'] / actions['sku_action_6']
        actions.to_csv(file, index= False)
        return actions
#--------------------------------------------------------------------------------
weekday_map = {
    '0' : 'mon',
    '1' : 'Tue',
    '2' : 'Wes',
    '3' : 'Thu',
    '4' : 'Fri',
    '5' : 'Sat',
    '6' : 'Sun'
}
def one_month_action_user_sku(start_date, end_date):
    file_user = '../cache/one_month_user_action_from_%s_to_%s.csv' % (start_date, end_date)
    file_sku= '../cache/one_month_sku_action_from_%s_to_%s.csv' % (start_date, end_date)
    if os.path.exists(file_user) and os.path.exists(file_sku):
        return pd.read_csv(file_user), pd.read_csv(file_sku)
    else:
        actions = get_actions(start_date, end_date)
        actions['weekday'] = actions['time'].map(lambda x: x.weekday()).astype(str)
        actions['weekday'] = actions['weekday'].map(weekday_map) + actions['type'].astype(str)
        weekday_user = pd.get_dummies(actions['weekday'],prefix ='user')
        weekday_sku = pd.get_dummies(actions['weekday'],prefix = 'sku')
        actions_user = pd.concat([actions['user_id'], weekday_user],axis=1)
        actions_sku = pd.concat([actions['sku_id'], weekday_sku],axis=1)
        actions_user = actions_user.groupby(['user_id'], as_index=False).sum()
        actions_sku= actions_sku.groupby(['sku_id'], as_index=False).sum()
        actions_user.to_csv(file_user,index=False)
        actions_sku.to_csv(file_sku,index=False)
        return actions_user, actions_sku

'''def one_month_action_user(start_date, end_date, daydelta):
    file_user = '../cache/one_month_user_action_from_%s_to_%s.csv' % (start_date, end_date)
    if os.path.exists(file_user) :
        return pd.read_csv(file_user)
    else:
        actions = get_actions(start_date, end_date)
        actions['weekday'] = actions['time'].map(lambda x: str(x.weekday()))
        actions['weekday'] = actions['weekday'].map(weekday_map) + actions['type'].astype(str)
        print('------>Gernearting  one_month_user_action_from_%s_to_%s.csv' % (start_date, end_date))
        weekday_user = pd.get_dummies(actions['weekday'],prefix = str(daydelta) +'th week user' )
        actions_user = pd.concat([actions['user_id'], weekday_user],axis=1)
        actions_user = actions_user.groupby(['user_id'], as_index=False).sum()
        actions_user.to_csv(file_user,index=False)
        return actions_user

def one_month_action_sku(start_date, end_date,daydelta):
    file_sku= '../cache/one_month_sku_action_from_%s_to_%s.csv' % (start_date, end_date)
    if os.path.exists(file_sku):
        return pd.read_csv(file_sku)
    else:
        actions = get_actions(start_date, end_date)
        actions['weekday'] = actions['time'].map(lambda x: str(x.weekday()))
        actions['weekday'] = actions['weekday'].map(weekday_map) + actions['type'].astype(str)

        print('------>Gernearting  one_month_user_action_from_%s_to_%s.csv' % (start_date, end_date))
        weekday_sku = pd.get_dummies(actions['weekday'],prefix =   str(daydelta) +'th week sku')
        actions_sku = pd.concat([actions['sku_id'], weekday_sku],axis=1)
        actions_sku= actions_sku.groupby(['sku_id'], as_index=False).sum()
        actions_sku.to_csv(file_sku,index=False)
        return actions_sku
#统计2个月内每个星期的点击行为
def everyweek_user_action():
        start_date = '2016-02-01'
        end_date = '2016-04-16'
        actions = None
        for i in (7, 14, 21, 28, 35, 42, 49, 56):
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')

            if actions is None:
                actions = one_month_action_user(start_days, end_date,i/7)
            else:
                actions = pd.merge(actions, one_month_action_user(start_days, end_date,i), how='left',
                                   on='user_id')
        return actions

def everyweek_sku_action():
        start_date = '2016-02-01'
        end_date = '2016-04-16'
        actions = None
        for i in (7, 14, 21, 28, 35, 42, 49, 56):
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')

            if actions is None:
                actions = one_month_action_sku(start_days, end_date,i/7)
            else:
                actions = pd.merge(actions, one_month_action_sku(start_days, end_date,i), how='left',
                                   on='sku_id')
        return actions '''
#----------------------------------------------------------------------------------------------------------------
def get_user_other_feat(end_date):
     file = '../cache/cache_get_user_other_feat_%s.csv'  % end_date
     if os.path.exists(file):
        return pd.read_csv(file)
     else:
        print('------>Gernearting  cache_get_user_other_feat_%s.csv' % (end_date))
        result = {}
        end_date = pd.datetime.strptime(end_date, '%Y-%m-%d')

        # 1.user 最后3天内交互天数
        start_date = end_date - pd.Timedelta('3 days')
        actions = _actions[
            (_actions.time >= start_date) & (_actions.time < end_date)
        ][['user_id', 'time']]
        actions.time = actions.time.apply(lambda x: x.day)
        actions = actions.drop_duplicates()
        actions = actions.groupby('user_id').size().reset_index(
            name='last3_active_days')
        result[1] = pd.concat([actions['user_id'],
                               pd.get_dummies(actions['last3_active_days'],
                                              prefix='last3_active_days')
                               ], axis=1)

        # 2.user 最后1周内交互天数
        start_date = end_date - pd.Timedelta('7 days')
        actions = _actions[
            (_actions.time >= start_date) & (_actions.time < end_date)
        ][['user_id', 'time']]
        actions.time = actions.time.apply(lambda x: x.day)
        actions = actions.drop_duplicates()
        actions = actions.groupby('user_id').size().reset_index(
            name='last7_active_days')
        result[2] = actions

        # 3.user 最后3周内交互天数
        start_date = end_date - pd.Timedelta('21 days')
        actions = _actions[
            (_actions.time >= start_date) & (_actions.time < end_date)
        ][['user_id', 'time']]
        actions.time = actions.time.apply(lambda x: x.day)
        actions = actions.drop_duplicates()
        actions = actions.groupby('user_id').size().reset_index(
            name='last21_active_days')
        result[3] = actions

        # 4.user 最后1次交互距离天数
        actions = _actions[(_actions.time < end_date)][['user_id', 'time']]
        actions = actions.groupby('user_id').time.max().reset_index(
            name='last_action_days')
        actions.last_action_days = (end_date - actions.last_action_days).apply(
            lambda x: x.days)
        result[4] = actions

        # 5. user 交互天数
        actions = _actions[(_actions.time < end_date)][['user_id', 'time']]
        actions.time = actions.time.apply(lambda x: x.month * 100 + x.day)
        actions = actions.drop_duplicates()
        actions = actions.groupby('user_id').size().reset_index(
            name='all_active_days')
        result[5] = actions

        # 6. user 交互最多的一天交互次数
        actions = _actions[(_actions.time < end_date)][['user_id', 'time']]
        actions.time = actions.time.apply(lambda x: x.month * 100 + x.day)
        actions = actions.groupby(['user_id', 'time']).size().reset_index(
            name='most_active_day_nums')
        actions = actions.groupby(
            'user_id', as_index=False)['most_active_day_nums'].max()
        result[6] = actions

        # 7. user 交互商品数
        actions = _actions[(_actions.time < end_date)][['user_id', 'sku_id']]
        actions = actions.drop_duplicates()
        actions = actions.groupby('user_id').size().reset_index(
            name='user_active_sku_id_num')
        result[7] = actions

        # 8.user 三天内购买商品数
        start_date = end_date - pd.Timedelta('3 days')
        actions = _actions[
            (_actions.time >= start_date) & (_actions.time < end_date)
        ][['user_id', 'type']]
        actions = actions[actions.type == 4]
        actions = actions.groupby(['user_id']).size().reset_index(
            name='last3_user_buy_num')
        result[8] = actions

        # 9.user 只有过一次交互的商品数
        actions = _actions[(_actions.time < end_date)][['user_id', 'sku_id']]
        actions = actions.groupby(['user_id', 'sku_id']).size()
        actions = actions[actions == 1].reset_index()
        actions = actions.groupby('user_id').size().reset_index(
            name='1_acti_sku_nums')
        result[9] = actions

        # 10.user 有过两次交互的商品数
        actions = _actions[(_actions.time < end_date)][['user_id', 'sku_id']]
        actions = actions.groupby(['user_id', 'sku_id']).size()
        actions = actions[actions == 2].reset_index()
        actions = actions.groupby('user_id').size().reset_index(
            name='2_acti_sku_nums')
        result[10] = actions

        # 11.user 有过三次交互的商品数
        actions = _actions[(_actions.time < end_date)][['user_id', 'sku_id']]
        actions = actions.groupby(['user_id', 'sku_id']).size()
        actions = actions[actions == 3].reset_index()
        actions = actions.groupby('user_id').size().reset_index(
            name='3_acti_sku_nums')
        result[11] = actions

        # 12.user 用户对每件商品的平均交互次数与方差
        actions = _actions[(_actions.time < end_date)][['user_id', 'sku_id']]
        actions = actions.groupby(['user_id', 'sku_id']).size().reset_index(
            name='acti_sku_nums')
        actions = actions[['user_id', 'acti_sku_nums']]
        grouped = actions.groupby('user_id', as_index=False)
        statis = pd.merge(grouped.mean(), grouped.var(),
                          how='left', on=['user_id'])
        actions = actions[['user_id']].drop_duplicates()
        actions = pd.merge(actions, statis, how='left', on=['user_id'])
        # result[12] = actions

        for i in result.values():
            actions = pd.merge(actions, i, how='left', on='user_id')
        actions.to_csv(file,index=False)
        return actions
#--------------------------------------------------------------------------------
def get_sku_other_feat(end_date):
     file = '../cache/cache_get_sku_other_feat_%s.csv'  % end_date
     if os.path.exists(file):
        return pd.read_csv(file)
     else:
        print('------>Gernearting  cache_get_sku_other_feat_%s.csv' % (end_date))
        result = {}
        end_date = pd.datetime.strptime(end_date, '%Y-%m-%d')

        # 1.sku 最后3天内交互天数
        start_date = end_date - pd.Timedelta('3 days')
        actions = _actions[
            (_actions.time >= start_date) & (_actions.time < end_date)
        ][['sku_id', 'time']]
        actions.time = actions.time.apply(lambda x: x.day)
        actions = actions.drop_duplicates()
        actions = actions.groupby('sku_id').size().reset_index(
            name='sku_last3_active_days')
        result[1] = pd.concat([actions['sku_id'],
                               pd.get_dummies(actions['sku_last3_active_days'],
                                              prefix='sku_last3_active_days')
                               ], axis=1)

        # 2.sku 最后1周内交互天数
        start_date = end_date - pd.Timedelta('7 days')
        actions = _actions[
            (_actions.time >= start_date) & (_actions.time < end_date)
        ][['sku_id', 'time']]
        actions.time = actions.time.apply(lambda x: x.day)
        actions = actions.drop_duplicates()
        actions = actions.groupby('sku_id').size().reset_index(
            name='sku_last7_active_days')
        result[2] = actions

        # 3.sku 最后3周内交互天数
        start_date = end_date - pd.Timedelta('21 days')
        actions = _actions[
            (_actions.time >= start_date) & (_actions.time < end_date)
        ][['sku_id', 'time']]
        actions.time = actions.time.apply(lambda x: x.day)
        actions = actions.drop_duplicates()
        actions = actions.groupby('sku_id').size().reset_index(
            name='sku_last21_active_days')
        result[3] = actions

        # 4.sku 最后1次交互距离天数
        actions = _actions[(_actions.time < end_date)][['sku_id', 'time']]
        actions = actions.groupby('sku_id').time.max().reset_index(
            name='sku_last_action_days')
        actions.sku_last_action_days = (end_date - actions.sku_last_action_days).apply(
            lambda x: x.days)
        result[4] = actions

        # 5. sku 交互天数
        actions = _actions[(_actions.time < end_date)][['sku_id', 'time']]
        actions.time = actions.time.apply(lambda x: x.month * 100 + x.day)
        actions = actions.drop_duplicates()
        actions = actions.groupby('sku_id').size().reset_index(
            name='sku_all_active_days')
        result[5] = actions

        # 6. sku 交互最多的一天交互次数
        actions = _actions[(_actions.time < end_date)][['sku_id', 'time']]
        actions.time = actions.time.apply(lambda x: x.month * 100 + x.day)
        actions = actions.groupby(['sku_id', 'time']).size().reset_index(
            name='sku_most_active_day_nums')
        actions = actions.groupby(
            'sku_id', as_index=False)['sku_most_active_day_nums'].max()
        result[6] = actions

        # 7. sku 交互user数
        actions = _actions[(_actions.time < end_date)][['user_id', 'sku_id']]
        actions = actions.drop_duplicates()
        actions = actions.groupby('sku_id').size().reset_index(
            name='sku_active_user_id_num')
        result[7] = actions

        # 8.sku三天内被购买数
        start_date = end_date - pd.Timedelta('3 days')
        actions = _actions[
            (_actions.time >= start_date) & (_actions.time < end_date)
        ][['sku_id', 'type']]
        actions = actions[actions.type == 4]
        actions = actions.groupby(['sku_id']).size().reset_index(
            name='last3_sku_buy_num')
        result[8] = actions

        # 9.与sku 只有过一次交互的用户数
        actions = _actions[(_actions.time < end_date)][['user_id', 'sku_id']]
        actions = actions.groupby(['user_id', 'sku_id']).size()
        actions = actions[actions == 1].reset_index()
        actions = actions.groupby('sku_id').size().reset_index(
            name='1_acti_user_nums')
        result[9] = actions

        # 10.与sku 只有过二次交互的用户数
        actions = _actions[(_actions.time < end_date)][['user_id', 'sku_id']]
        actions = actions.groupby(['user_id', 'sku_id']).size()
        actions = actions[actions == 2].reset_index()
        actions = actions.groupby('sku_id').size().reset_index(
            name='2_acti_user_nums')
        result[10] = actions

        # 11.与sku 只有过三次交互的用户数
        actions = _actions[(_actions.time < end_date)][['user_id', 'sku_id']]
        actions = actions.groupby(['user_id', 'sku_id']).size()
        actions = actions[actions == 3].reset_index()
        actions = actions.groupby('sku_id').size().reset_index(
            name='3_acti_user_nums')
        result[11] = actions

        # 12.sku 对于user的平均交互次数与方差
        actions = _actions[(_actions.time < end_date)][['user_id', 'sku_id']]
        actions = actions.groupby(['user_id', 'sku_id']).size().reset_index(
            name='acti_user_nums')
        actions = actions[['sku_id', 'acti_user_nums']]
        grouped = actions.groupby('sku_id', as_index=False)
        statis = pd.merge(grouped.mean(), grouped.var(),
                          how='left', on=['sku_id'])
        actions = actions[['sku_id']].drop_duplicates()
        actions = pd.merge(actions, statis, how='left', on=['sku_id'])
        # result[12] = actions

        for i in result.values():
            actions = pd.merge(actions, i, how='left', on='sku_id')
        actions.to_csv(file,index=False)
        return actions
#--------------------------------------------------------------------------------
def get_user_brand(end_date):
     file = '../cache/cache_get_user_brand_%s.csv'  % end_date
     if os.path.exists(file):
        return pd.read_csv(file)
     else:
        print('------>Gernearting  cache_get_user_brand_%s.csv' % (end_date))
        result = {}
        end_date = pd.datetime.strptime(end_date, '%Y-%m-%d')

        #1.user 最后3天内交互（1-6）品牌平均数
        start_date = end_date - pd.Timedelta('3 days')
        actions = _actions[(_actions.time >= start_date) & (_actions.time < end_date)][['user_id', 'brand']]
        actions = pd.get_dummies(actions,columns=['brand'])
        actions = (actions.groupby('user_id').sum()/3).reset_index('user_id')
        result[1] = actions

        #2.user 最后7天内交互（1-6）品牌平均数
        start_date = end_date - pd.Timedelta('7 days')
        actions = _actions[(_actions.time >= start_date) & (_actions.time < end_date)][['user_id', 'brand']]
        actions = pd.get_dummies(actions,columns=['brand'])
        actions = (actions.groupby('user_id').sum()/7).reset_index('user_id')
        result[2] = actions

        #3.user 最后21天内交互（1-6）品牌平均数
        start_date = end_date - pd.Timedelta('21 days')
        actions = _actions[(_actions.time >= start_date) & (_actions.time < end_date)][['user_id', 'brand']]
        actions = pd.get_dummies(actions,columns=['brand'])
        actions = (actions.groupby('user_id').sum()/21).reset_index('user_id')
        result[3] = actions

        #4.user 总时间内交互（1-6）品牌平均数与方差
        actions = _actions[(_actions.time < end_date)][['user_id', 'brand']]
        actions = pd.get_dummies(actions,columns=['brand'])
        grouped = actions.groupby('user_id', as_index=False)
        statis = pd.merge(grouped.mean(), grouped.var(),how='left', on=['user_id'])
        actions = pd.merge(actions[['user_id']], statis,how='left', on=['user_id'])
        actions = actions.drop_duplicates()
        result[4] = actions

        #5.user 交互最多的一天交互总品牌数
        actions = _actions[(_actions.time < end_date)][['user_id', 'brand']]
        actions = actions.groupby(['user_id', 'brand']).size().reset_index(name='most_active_day_brand')
        actions = actions.groupby('user_id', as_index=False)['most_active_day_brand'].max()
        result[5] = actions

        #6.user 只有过一次交互的总品牌数
        actions = _actions[(_actions.time < end_date)][['user_id', 'brand']]
        actions = actions.groupby(['user_id', 'brand']).size()
        actions = actions[actions == 1].reset_index()
        actions = actions.groupby('user_id').size().reset_index(name='1_acti_brand_nums')
        result[6] = actions

        #7.user 有过两次交互的总品牌数
        actions = _actions[(_actions.time < end_date)][['user_id', 'brand']]
        actions = actions.groupby(['user_id', 'brand']).size()
        actions = actions[actions == 2].reset_index()
        actions = actions.groupby('user_id').size().reset_index(name='2_acti_brand_nums')
        result[7] = actions

        #8.user 有过三次交互的总品牌数
        actions = _actions[(_actions.time < end_date)][['user_id', 'brand']]
        actions = actions.groupby(['user_id', 'brand']).size()
        actions = actions[actions == 3].reset_index()
        actions = actions.groupby('user_id').size().reset_index(name='3_acti_brand_nums')
        result[8] = actions

        #9.user 用户对总品牌数的平均交互次数与方差
        actions = _actions[(_actions.time < end_date)][['user_id', 'brand']]
        actions = actions.groupby(['user_id', 'brand']).size().reset_index(name='acti_brand_nums')
        actions = actions[['user_id', 'acti_brand_nums']]
        grouped = actions.groupby('user_id', as_index=False)
        statis = pd.merge(grouped.mean(), grouped.var(),
                                  how='left', on=['user_id'])
        actions = actions[['user_id']].drop_duplicates()
        actions = pd.merge(actions[['user_id']], statis,
                                   how='left', on=['user_id'])

        for i in result.values():
            actions = pd.merge(actions, i, how='left', on='user_id')
        actions.to_csv(file,index=False)
        return actions
#-----------------------------------------------------------------------------
def get_weekday_mean_var():
    end_date = '2016-04-16'
    file = '../cache/cache_get_weekday_mean_var_%s.csv' % end_date
    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        result = {}
        end_date = pd.datetime.strptime(end_date, '%Y-%m-%d')
        actions = _actions[(_actions.time < end_date)]
        actions['weekday'] = actions['time'].map(lambda x: x.weekday())
        actions = actions[['user_id','weekday','type']]
        for i in range(0,7):
            statis  = actions[actions['weekday'] == i].groupby('user_id',as_index=False)
            statis = pd.merge(statis.mean(), statis.var(),
                                          how='left', on=['user_id'])
            statis = statis[['user_id','type_x','type_y']]
            statis.columns=['user_id','%s_day_mean'%i,'%s_day_var'%i]
            result[i] = statis
        actions = actions[['user_id']].drop_duplicates()
        for i in result.values():
            actions = pd.merge(actions, i, how='left', on=['user_id'])
        actions.to_csv(file,index=False)
        return actions
#----------------------------------------------------------------------------
def get_labels(start_date, end_date):
     file = '../cache/cache_labels_from_%s_to_%s.csv' % (start_date, end_date)
     if os.path.exists(file):
        return pd.read_csv(file)
     else:
        print('------>Gernearting  cache_labels_from_%s_to_%s.csv' % (start_date, end_date))
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type'] == 4]
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        # 标记成功交易的用户与商品
        actions = actions[['user_id', 'sku_id', 'label']]
        
        actions.to_csv(file, index=False)
        return actions


# -------------------------------------------------------------------
def make_test_set(train_start_date, train_end_date):
    file = '../cache/cache_test_set_%s_%s.csv' % (train_start_date, train_end_date)
    if os.path.exists(file):
        actions = pd.read_csv(file)
    else:
        print('------>Gernearting  cache_test_set_%s_to_%s.csv' % (train_start_date, train_end_date))
        start_days = "2016-02-01"
        user = get_adv_user_feat(train_start_date, train_end_date)
        product = get_basic_product_feat()
        user_acc = one_month_user_action_mean_var(start_days, train_end_date)
        product_acc = one_month_sku_action_mean_var(start_days, train_end_date)
        comment_acc = get_comments_product_feat(train_start_date, train_end_date)
        action_user,  actions_sku= one_month_action_user_sku(start_days,train_end_date)
        user_other_fea = get_user_other_feat(train_end_date)
        sku_other_fea = get_sku_other_feat(train_end_date)
        user_brand = get_user_brand(train_end_date)
        weekday_static = get_weekday_mean_var()
        #labels = get_labels(test_start_date, test_end_date)

        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        actions = None
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            if actions is None:
                actions = get_action_feat(start_days, train_end_date)
            else:
                actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                                   on=['user_id', 'sku_id'])
        actions = pd.merge(actions, user, how='left', on='user_id')#112
        actions = pd.merge(actions, user_acc, how='left', on='user_id')#23
        actions = pd.merge(actions, product, how='left', on='sku_id')#112
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')#23
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')#6
        actions = pd.merge(actions, action_user, how = 'left', on='user_id')#42
        actions = pd.merge(actions, actions_sku, how = 'left', on='sku_id')#42
        actions = pd.merge(actions, user_other_fea, how='left', on='user_id')
        actions = pd.merge(actions, sku_other_fea, how='left', on='sku_id')
        actions = pd.merge(actions, user_brand, how='left', on='user_id')
        actions = pd.merge(actions, weekday_static, how='left', on='user_id')#14
        #actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
        actions = actions.fillna(0)
        actions = actions[actions['cate'] == 8]
        actions.to_csv(file, index=False)

    users = actions[['user_id', 'sku_id']].copy()
    del actions['user_id']
    del actions['sku_id']
    return users, actions

def make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, days=30):
    print('------>Gernearting  train_set_%s_%s_%s_%s.csv' % (train_start_date, train_end_date, test_start_date, test_end_date))
    start_days = "2016-02-01"
    user = get_adv_user_feat(train_start_date, train_end_date) # 基本用户信息，用户ID，年龄，性别，用户等级 (未使用: 用户注册日期)....
    product = get_basic_product_feat() # 基本商品信息，商品编号，属性1，属性2，属性3，品类ID，品牌ID....
    user_acc = one_month_user_action_mean_var(start_days, train_end_date)
    product_acc = one_month_sku_action_mean_var(start_days, train_end_date)
    comment_acc = get_comments_product_feat(train_start_date, train_end_date)
    action_user,  actions_sku= one_month_action_user_sku(start_days,train_end_date)
    user_other_fea = get_user_other_feat(train_end_date)
    sku_other_fea = get_sku_other_feat(train_end_date)
    user_brand = get_user_brand(train_end_date)
    weekday_static = get_weekday_mean_var()
    labels = get_labels(test_start_date, test_end_date)

    # generate 时间窗口
    # actions = get_accumulate_action_feat(train_start_date, train_end_date)
    actions = None
    for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_action_feat(start_days, train_end_date)
        else:
            actions = pd.merge(actions, get_action_feat(start_days, train_end_date), how='left',
                               on=['user_id', 'sku_id'])

    actions = pd.merge(actions, user, how='left', on='user_id')
    actions = pd.merge(actions, user_acc, how='left', on='user_id')
    actions = pd.merge(actions, product, how='left', on='sku_id')
    actions = pd.merge(actions, product_acc, how='left', on='sku_id')
    actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
    actions = pd.merge(actions, action_user, how = 'left', on='user_id')
    actions = pd.merge(actions, actions_sku, how = 'left', on='sku_id')
    actions = pd.merge(actions, user_other_fea, how='left', on='user_id')
    actions = pd.merge(actions, sku_other_fea, how='left', on='sku_id')
    actions = pd.merge(actions, user_brand, how='left', on='user_id')
    actions = pd.merge(actions, weekday_static, how='left', on='user_id')
    actions = pd.merge(actions, labels, how='left', on=['user_id', 'sku_id'])
    actions = actions.fillna(0)
    dump_path = '../cache/train_set_%s_%s_%s_%s.csv' % (train_start_date, train_end_date, test_start_date, test_end_date)
    actions.to_csv(dump_path, index=False)

    users = actions[['user_id', 'sku_id']].copy()
    labels = actions['label'].copy()
    del actions['user_id']
    del actions['sku_id']
    del actions['label']

    return users, actions, labels


def report(pred, label):

    actions = label
    result = pred

    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()

    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / ( pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
    print('所有用户中预测购买用户的召回率' + str(all_user_recall))

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
    print('所有用户中预测购买商品的召回率' + str(all_item_recall))
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print('F11=' + str(F11))
    print('F12=' + str(F12))
    print('score=' + str(score))


if __name__ == '__main__':
    train_start_date = '2016-02-01'
    train_end_date = '2016-03-01'
    test_start_date = '2016-03-01'
    test_end_date = '2016-03-05'
    user, action, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    print(user.head(10))
    print(action.head(10))
