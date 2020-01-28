# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:27:21 2020

@author: Vsolanki
"""

if __name__ == '__main__':

    import sys
    import pandas as pd
    import pandas_profiling
    from flask import Flask, request,jsonify , send_file
    from flask import Flask
    import matplotlib
    import Recurrence_service

    
    matplotlib.use('Agg')

    global dataframe_recurrence

    APP = Flask(import_name="recurrence")

    @APP.route('/recurrence', methods=['POST'])
    def upload_file():
        global input_file
        # I used form data type which means there is a
        # "Content-Type: application/x-www-form-urlencoded"
        # header in my request
        # raw_data = request.files['myfile'].read()
        # In form data, I used "myfile" as key.
        try:
            file = request.files['file']
            data = pd.read_excel(file, encoding='UTF-8')
            dataframe_recurrence = Recurrence_service.recurrence(data)
            dataframe_recurrence.to_excel("dataframe_recurrence.xlsx", index=False)
             
            return send_file("dataframe_recurrence.xlsx", attachment_filename='dataframe_recurrence.xlsx')
        except Exception as error:
            return jsonify({'msg': "File Upload Error", 'exception':error}), 401


    @APP.route("/get_matches", methods=['GET'])

    def get_insights():

        input_sentence = request.args.get('input_sentence')
        dataframe_output =  Recurrence_service.similar_search(input_sentence)
        dataframe_output.to_excel("dataframe_output.xlsx", index=False)

        return  send_file("dataframe_output.xlsx", attachment_filename='dataframe_output.xlsx')

        # profile = data_xls.profile_report()
        # profile = data_xls.profile_report(title='Pandas Profiling Report')
        # ht=profile.to_file(output_file="output.html")
        # with open(ht, 'r') as f:
        # html1 = f.read()




    APP.run(host="10.133.6.204", port="5002")
