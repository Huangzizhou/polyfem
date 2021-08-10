# %%
import plotly.graph_objects as go
import numpy as np

# %%
with open("data.csv",'r') as f:
    lines = f.readlines()
    lines = [line[:-1].split(",") for line in lines[1:]]

for j in range(len(lines)):
    line = lines[j]
    lines[j][0] = float(line[0])
    lines[j][-1] = float(line[-1])
    for i in range(1, 4):
        lines[j][i] = int(line[i])

# print(lines)
def get_class_name(line):
    return "visc"+str(line[0])+"_vOrder"+str(line[2])+"_pOrder"+str(line[3])

datas = []
names = []
for line in lines:
    name = get_class_name(line)
    try:
        index = names.index(name)
        datas[index].append([0.2 / 2**line[1], line[-1]])
    except:
        names.append(name)
        datas.append([[0.2 / 2**line[1], line[-1]]])

plots = []
for data, name in zip(datas, names):
    data = np.array(data)
    rate = abs(np.polyfit(np.log2(data[:,0]), np.log2(data[:,1]), 1)[0])
    print(name, rate)
    plots.append(go.Scatter(x = data[:,0], y = data[:,1], mode = 'lines+markers', 
            name = name + ": " + str(rate)))


# # %%
layout = go.Layout(
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    ),
    xaxis=dict(
        title="h",
        tickformat='.1e',
        exponentformat='power',
        type='log',
        nticks=5,
        tickfont=dict(
            size=10
        ),
    ),
    yaxis=dict(
        title="L2 error",
        tickformat='.1e',
        exponentformat='power',
        type='log',
        tickfont=dict(
            size=10
        ),
    ),
    font=dict(
        size=12
    )
)

fig = go.Figure(data=plots, layout=layout)
draft_template = go.layout.Template()
fig.update_layout(template=draft_template)
# fig.show()
fig.write_image("data.png", scale=3)
