from flask import Flask, render_template,request
import pickle
import numpy as np
import math

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def man():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    chestsize=math.ceil(pred[0][0]/25.4)
    shouldersize=math.ceil(pred[0][1]/25.4)
    Size=""
    if (chestsize<32):
        if(shouldersize<40):
            Size= "Extra Small"
        else:
            Size= "Small"
    elif (chestsize>=32 and chestsize<36):
        if(shouldersize<45):
            Size= "Small"
        else:
            Size= "Medium"
    elif (chestsize>=36 and chestsize<40):
        if(shouldersize<55):
            Size= "Medium"
        else:
            Size= "Large"
    elif (chestsize>=40 and chestsize<44):
        if(shouldersize<60):
            Size= "Large"
        else:
            Size= "Extra Large"
        
    elif (chestsize>=44 and chestsize<48):
        if(shouldersize<65):
            Size= "Extra Large"
        else:
            Size= "2-Extra Large"
    elif (chestsize>=48):
        Size= "2-Extra Large"
    Size="Your Predicted T shirt Size: \n"+Size

    return render_template('home.html', tsize=Size)


if __name__ == "__main__":
    app.run()
