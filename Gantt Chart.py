import plotly.plotly as py
import plotly.figure_factory as ff

df = [dict(Task="Learn Theory Prediction", Start='2019-04-30', Finish='2019-06-27', Complete=0),
      dict(Task="Defining Prediction", Start='2019-05-07', Finish='2019-05-21', Complete=0),
      dict(Task="Planning Prediction", Start='2019-05-12', Finish='2019-05-21', Complete=0),
      dict(Task="Implementing Prediction", Start='2019-05-21', Finish='2019-06-24', Complete=0),
      dict(Task="Testing Prediction", Start='2019-06-20', Finish='2019-6-29', Complete=0),
      dict(Task="Checking Prediction", Start='2019-06-28', Finish='2019-06-30', Complete=0)]

fig = ff.create_gantt(df, colors='Viridis', index_col='Complete', show_colorbar=True)
py.iplot(fig, filename='machine-learning-initial-gantt', world_readable=True)
