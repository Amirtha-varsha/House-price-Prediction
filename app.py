from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
application=Flask(__name__)

app=application
#route for home page
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictdate',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            total_sqft=float(request.form.get('total_sqft')),
            bath=int(request.form.get('bath')),
            bhk=request.form.get('bhk'),
            location=request.form.get('location')

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        # Format the predicted value (rounding to two decimal places)
        formatted_results = "{:,.2f}".format(results[0])  # Assuming results[0] is the predicted value
        print(f"Prediction: {results[0]}")
        return render_template('home.html',results=formatted_results)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)