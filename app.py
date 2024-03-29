


from flask import Flask, render_template, request, jsonify
import pickle





app = Flask(__name__)


# In[4]:


# Load scaler and model
with open('diabeticscaler.pkl', 'rb') as scaler_file:
    diabitic_scaler = pickle.load(scaler_file)

with open('diabeticmodel.pkl', 'rb') as model_file:
    diabitic_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('diabeticmodel.html')

@app.route('/diabiticmodelpredict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        scaled_features = diabitic_scaler.transform([features])
        prediction = diabitic_model.predict(scaled_features)

        result = None
        if prediction[0] == 1:
            result = 'The patient is diabetic.'
        else:
            result = 'The patient is not diabetic.'

        return render_template('diabeticmodel.html', prediction=result)


# In[ ]:


if __name__ == '__main__':
    app.run(host='0.0.0.0')

