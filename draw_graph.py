from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

def drawGraph(number_of_epochs, training_loss, training_accuracy, testing_accuracy):
    py.sign_in('VikramShenoy','x1Un4yD3HDRT838vRkFA')
    x_axis = []
    y_axis = []
    for i in range(0, number_of_epochs):
        if(i%20 == 0):
            y_axis.append(float(training_loss[i/20]))
            x_axis.append(i)

    trace0 = go.Scatter(
    x = x_axis,
    y = y_axis,
    mode = "lines",
    name = "Cost"
    )
    data = go.Data([trace0])
    layout = go.Layout()
    fig = go.Figure(data=data, layout=layout)
    fig['layout']['xaxis'].update(title="Number of Epochs", range = [min(x_axis), max(x_axis)], dtick = 100, showline = True)
    fig['layout']['yaxis'].update(title="Loss")
    py.image.save_as(fig, filename="Graphs/Loss_Graph.png")

    print "Loss Graph Created"
    x_axis = ["Training", "Testing"]
    y_axis = [training_accuracy, testing_accuracy]


    trace1 = go.Bar(
    x = x_axis,
    y = y_axis,
    width = [0.3, 0.3]
    )
    data = [trace1]
    layout = go.Layout()
    fig = go.Figure(data=data, layout=layout)
    fig['layout']['xaxis'].update(title="Mode", showline = True)
    fig['layout']['yaxis'].update(title="Accuracy")
    py.image.save_as(fig, filename="Graphs/Accuracy_Graph.png")
    print "Accuracy Graph Created"
