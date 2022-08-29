from flask import Flask, render_template, request, flash, redirect, url_for
import pickle, os, pandas as pd

MODEL_PATH = os.path.join(os.getcwd(), 'model.pkl') 
model_name = "Climbing Accident Severity Prediciton"
version = "v1.0.0"

month_cols = ['January','February', 'March', 'April', 'May', 'June', 'July', 'August','September', 'October', 'November', 'December']
exp_cols = ['No/Little', 'Moderate','Experienced', 'Unknown']
climbing_type_cols = ['Descent', 'Roped', 'Trad Climbing', 'Sport','Top-Rope', 'Aid & Big Wall Climbing', 'Pendulum', 'Unroped ', 'Solo','Climbing Alone', 'Bouldering', 'Non-climbing','Alpine/Mountaineering']
alpine_ice_factors_cols = ['Piton/Ice Screw', 'Ascent Illness', 'Crampon Issues', 'Ice Climbing', 'Glissading', 'Ski-related ', 'Poor Position']
natural_factors_cols =['Poor Cond/Seasonal Risk', 'Avalanche','Cornice / Snow Bridge Collapse', 'Bergschrund','Crevasse / Moat / Berschrund', 'Icefall / Serac / Ice Avalanche',
       'Exposure', 'Non-Ascent Illness', 'Visibility', 'Severe Weather','Wildlife', 'Natural Rockfall']
human_errors=['Off-route', 'Rushed', 'Run Out','Crowds', 'Inadequate Food/Water', 'No Helmet', 'Late in Day',
       'Late Start', 'Party Separated', 'Ledge Fall', 'Gym / Artificial','Gym Climber', 'Fatigue', 'Large Group', 'Distracted',
       'Object Dropped/Dislodged', 'Handhold/Foothold Broke','Knot & Tie-in Error', 'No Backup or End Knot', 'Gear Broke',
       'Intoxicated', 'Inadequate Equipment', 'Inadequate Protection / Pulled','Anchor Failure / Error', 'Stranded / Lost / Overdue', 'Belay Error',
       'Rappel Error', 'Lowering Error', 'Miscommunication']

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/model', methods=['GET', 'POST'])
def model():
    if request.method == 'POST':
        
        exp = request.form.get('exp')
        ct_check_list = request.form.getlist('ct')
        ai_check_list = request.form.getlist('ai')
        nt_check_list = request.form.getlist('nt')
        month = request.form.get('month')

        input_dt = [2022]
        for i in range(77):
            input_dt.append(0)
        
        input_pd = pd.DataFrame ([input_dt],  columns = ['Publication Year']+exp_cols+climbing_type_cols+alpine_ice_factors_cols+natural_factors_cols+human_errors+month_cols)
        
        for i in range(len(input_pd.columns)):
            if exp == input_pd.iloc[:,i].name:
                input_pd.iloc[:, i] = 1
            if month == input_pd.iloc[:,i].name:
                input_pd.iloc[:, i] = 1
            for item in ct_check_list:
                if item == input_pd.iloc[:,i].name:
                    input_pd.iloc[:, i] = 1
            for item in ai_check_list:
                if item == input_pd.iloc[:,i].name:
                    input_pd.iloc[:, i] = 1
            for item in nt_check_list:
                if item == input_pd.iloc[:,i].name:
                    input_pd.iloc[:, i] = 1

        with open('model.pkl','rb') as f:
            model = pickle.load(f)
        pred =model.predict(input_pd)[0]

        return render_template('prediction.html', pred = pred)
    
    else:
        return render_template('model.html')

if __name__ == "__main__":
    app.run()