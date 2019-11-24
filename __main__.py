import dask
import dask.dataframe as dd
from scipy.stats import t
from scipy.stats import ttest_ind_from_stats as t_test
import math
# Для корректной работы необходима библиотека dask
# dask решает проблему недостаточности памяти машины для вмещения всех данных и автоматически разбивает данные на части и распараллеливает процесс обработки
# dask поддерживает pandas-like формат манипуляции данными



# Путь до данных
file_path = "/usr/local/data/transactions.txt"

# Считываем данные, при этом даём свои наменования столбцам следующим образом
# TrancNum -- номер транкзации
# ClientID -- ID клиента
# VolumeRUR -- объем транкзации в рублях
# Segment   -- сегмент
db=dd.read_csv(file_path, names = ['TrancNum','ClientID','VolumeRUR','Segment'])


# Задаем задачи обработки файла для dask
# Внимание! До запуска метода dask.compute() данные ещё не будут доступны.

SegmentsUniqClient = db.groupby('Segment').ClientID.nunique()           # Содержит разбитые по сегментам R и AF количества уникальных клиентов, совершивших транкзацию
MeanVolume = db.groupby('Segment').VolumeRUR.mean()                     # Содержит разбитые по сегментам R и AF средние объемы одной транкзации
StdVolume = db.groupby('Segment').VolumeRUR.std()                       # Содержит разбитые по сегментам R и AF стандартные отклонения для объема транкзации
SegmentsTotalTranc = db.groupby('Segment').TrancNum.count()             # Содержит разбитые по сегментам R и AF суммарные количества транкзации


# Вызывается метот dask.compute(). Машина ничинает обработку данных
SegmentsUniqClient,MeanVolume,StdVolume,SegmentsTotalTranc = dask.compute(SegmentsUniqClient,MeanVolume,StdVolume,SegmentsTotalTranc)




# По сегментам R и AF оздаются удобные для оперирования переменные
R_mean = MeanVolume["R"]
R_std = StdVolume['R']
R_num = SegmentsTotalTranc['R']

AF_mean = MeanVolume["AF"]
AF_std = StdVolume['AF']
AF_num = SegmentsTotalTranc['AF']



# Вычисление коррекций среднего для каждого сегмента для получения доверительных интервалов 90%
Del_R = t.ppf(0.05,R_num-1)*R_std/math.sqrt(R_num)
Del_AF = t.ppf(0.05,AF_num-1)*AF_std/math.sqrt(AF_num)

# (R_interval_left, R_interval_right ) -- искомый доверительный интервал для сегмента R
R_interval_left = R_mean - Del_R
R_interval_right = R_mean + Del_R

# (AF_interval_left, AF_interval_right ) -- искомый доверительный интервал для сегмента AF
AF_interval_left = AF_mean - Del_AF
AF_interval_right = AF_mean + Del_AF


# Тестирование гипотезы об раменстве средних объёмов по сегментам для уровня значимости 0.1
Test_result_statistic,Test_result_pval  = t_test(R_mean,R_std,R_num,AF_mean,AF_std,AF_num)


Hypothesis_is_accepted = Test_result_pval < 0.1


# Если Hypothesis_is_accepted == True -- гипотеза принята, иначе гипотеза отвергнута


