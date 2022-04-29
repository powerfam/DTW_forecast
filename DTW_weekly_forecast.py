#!/usr/bin/env python
# coding: utf-8

# In[2]:


def dtw_weekly(df, start_date, end_date, sear_start_year, sear_start_month, sear_end_year, sear_end_month, forecast=True):
    
    """
    Args:
        df: - 인덱스가 '2022-01-01'과 같은 시계열인 데이터프레임 필요
              
        start_date: - 자기가 패턴을 파악하고 싶은 구간 시작일 입력
        
        end_date: - 자기가 패턴을 파악하고 싶은 구간 종료일 입력
        
        sear_start_year: - 해당 패턴을 찾고 싶은 과거 기간 시작 년도 입력
        
        sear_start_month: - 해당 패턴을 찾고 싶은 과거 기간 시작 월 입력
        
        sear_end_year: - 해당 패턴을 찾고 싶은 과거 기간 종료 년도 입력
        
        sear_end_month: - 해당 패턴을 찾고 싶은 과거 기간 종료 월 입력
        
        forecast: True / False - 유사한 패턴 시기의 향후 +12 시점 이후(ex-주간 데이터:향후 12주 흐름) 가격 도출
                    
    
    Returns:
        1. Top 30 매칭 구간 추출
        2. Top 3 매칭 구간 시각화 
        3. 향후 12주 흐름 도출
        4. Top 30 매칭 구간을 활용한 향후 12기간-1기간 가격 변동률 분포
        
    """
    # 필요 패키지 로드
    
    import numpy as np
    import pandas as pd  
    import fastdtw.fastdtw as fastdtw
    
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rc
    font_path = "C:/Windows/Fonts/NGULIM.TTF"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
    
    from datetime import datetime as dt
    from dateutil.relativedelta import relativedelta  
    
    import warnings 
    warnings.filterwarnings('ignore')

    
    
    # 데이터 스케일링
    real_df = df
    real_df['yoy_data']= df['data'].pct_change(12)
    real_df['wow_data']= df['data'].pct_change(1)
    df = pd.DataFrame(real_df['yoy_data'].dropna())
    
    # 중심 패턴 기간 데이터 불러오기
    condition = (df.index >= start_date) & (df.index <= end_date)
    target_df = df.loc[condition, :]
    target_length = len(target_df)
        
    # 과거 db 구간 찾기 
    search_df = df.copy()
    start_date=f"{sear_start_year}-{str(sear_start_month).zfill(2)}" 
    end_date=f"{sear_end_year}-{sear_end_month}"
    search_df = search_df.loc[start_date:end_date, :]
    
    # 과거 db 구간 전체 대상으로 동일한 길이 만큼 데이터 불러오기 
    # 그 이후 데이터셋들을 search_df_sets에 저장
    search_df_sets = []
    search_df_sets_idx = []
    for i in range(0, len(search_df)-target_length+1):
        tmp = search_df.iloc[i:i+target_length]
        search_df_sets.append(tmp)
        search_df_sets_idx.append(tmp.index)
        
    #### 모든 데이터 셋에 대해서 DTW 거리 계산 ###
    tmp_distance=[]
    tmp_path=[]
    idx_list = []
    idx = 0
    for sear in search_df_sets:
        d, p = fastdtw(sear, target_df)
        tmp_distance.append(d)
        tmp_path.append(p)
        idx_list.append(idx)
        idx += 1
        
    
    # 거리가 작은 순서대로 정렬, 인덱스는 search_df_sets_index를 출력하기 위해서 활용
    distance_df = pd.DataFrame(tmp_distance, columns=['distance'])
    idx_df = pd.DataFrame(idx_list, columns=['index'])
    distance_df = distance_df.join(idx_df, how='inner').sort_values(by='distance', ascending=True)
    
    
    
    ### top 30 period 기간 출력 ###
    periods=[]
    best_periods_idx = []
    top_period_name = []
    for t in range(0, 30):
        best_period = distance_df.index[t]
        best_idx = search_df_sets_idx[best_period]
        best_list = [best_idx[0], best_idx[-1]]
        best_periods_idx.append(best_idx)
        periods.append(best_list)
        
    print('가장 유사한 기간 top 10 리스트')
    for r in range(1,11):       
        print(f'{r}위:', periods[r-1][0],'~',periods[r-1][1])
        name = str(periods[r-1][0].year)+ str(periods[r-1][0].month).zfill(2) + str(periods[r-1][0].day).zfill(2)+ '~' + str(periods[r-1][1].year)+str(periods[r-1][1].month).zfill(2)+str(periods[r-1][1].day).zfill(2)
        top_period_name.append(name)
        
    ### 시각화 ###
    first_cond = best_periods_idx[0]
    second_cond = best_periods_idx[1]
    third_cond = best_periods_idx[2]
    plt.figure(figsize=(8, 4))
    plt.plot(real_df.loc[target_df.index, 'yoy_data'].values, color='black', label='target_real', alpha=0.75)
    plt.plot(real_df.loc[first_cond, 'yoy_data'].values, color='r', label=top_period_name[0], alpha=0.75)
    plt.plot(real_df.loc[second_cond, 'yoy_data'].values, color='g', label=top_period_name[1], alpha=0.75)        
    plt.plot(real_df.loc[third_cond, 'yoy_data'].values, color='b', label=top_period_name[2], alpha=0.75)        
    
    plt.legend()
    plt.title("target vs. similar_periods", fontsize=15)
    plt.xlabel("주차")
    plt.ylabel("yoy, %")
    plt.show()
    
    
    
    ### 예측 ###
    
    # 예측치 산출을 위해서 top30의 12길이 만큼의 미래 시점 도출 (for index) 
    # 일단 12는 고정변수고, 나중에 12로 처리 가능 (j in range(1, 13))
    if forecast:
        forecast_top10_index_list = []
        
        for i in range(0, 30):
                if i == 0:
                    specific_idx_list = []
                    for j in range(1, 13):
                        weeks = j
                        forecast_index = best_periods_idx[i][-1]+relativedelta(weeks=weeks)
                        specific_idx_list.append(forecast_index)
                    forecast_top10_index_list.append(specific_idx_list) 
                   
                elif i != 0:
                    
                    specific_idx_list = []
                    for j in range(1, 13):
                        weeks = j
                        forecast_index = best_periods_idx[i][-1]+relativedelta(weeks=weeks)
                        specific_idx_list.append(forecast_index)
                    forecast_top10_index_list.append(specific_idx_list) 
    
    # 예측치 산출
    fcst_price_total = []
    for t in range(0, len(forecast_top10_index_list)):
        if t == 0:
            fcst_price_list = []
            start_price = real_df.loc[target_df.index, 'data'][-1]
            for cal_wow in forecast_top10_index_list[t]:
                cal_price = start_price * (1+real_df.loc[cal_wow, 'wow_data'])
                fcst_price_list.append(int(cal_price))
                start_price = cal_price
            fcst_price_total.append(fcst_price_list)
        
        elif t != 0:
            fcst_price_list = []
            start_price = real_df.loc[target_df.index, 'data'][-1]
            for cal_wow in forecast_top10_index_list[t]:
                cal_price = start_price * (1+real_df.loc[cal_wow, 'wow_data'])
                fcst_price_list.append(int(cal_price))
                start_price = cal_price
            fcst_price_total.append(fcst_price_list)
    
    # fcst_price_total[0] 은 top1 매칭 기간의 향후 가격 주간변동률을 고려한 흐름
    tmp = []
    for r in range(0, len(fcst_price_total)):
        tmp.append(pd.DataFrame(fcst_price_total[r], columns =[str('top') + str(r+1) + str(' ') + str('forecast')]))
        
    forecast_df = tmp[0]
    for k in range(1, len(fcst_price_total)):
        forecast_df = forecast_df.join(tmp[k], how='outer')
        
    
    vol_list = []
    for k in range(0, len(forecast_df_test.columns)):
        t_1 = forecast_df_test.iloc[0,k]
        t_n = forecast_df_test.iloc[-1,k]
        vol = ((t_n - t_1)/t_n)*100
        vol_list.append(vol)

    plt.hist(vol_list, bins=20)
    plt.title('top30 매칭기간 향후 12시점 대비 변동률 분포도')
    plt.grid(True)
    plt.xlabel('변동률(%)')
    plt.ylabel('갯수')
    plt.show()
    
    
    return {'forecast': forecast_df, 'top_period':top_period_name, 'top_period_index': best_periods_idx}
   

