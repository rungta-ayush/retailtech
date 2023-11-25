import json

from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Replace this URL with the actual Mindee API endpoint
MINDEE_API_ENDPOINT = 'https://api.mindee.net/v1/products/mindee/expense_receipts/v5/predict'

# Replace this with your Mindee API token
MINDEE_API_TOKEN = 'ab988a6386f4760a1c7ed9866c66f7c0'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Upload the file to the Mindee API
    files = {'document': (file.filename, file.read())}
    headers = {
        'Authorization': f'Token {MINDEE_API_TOKEN}'
    }

    response = requests.post(MINDEE_API_ENDPOINT, files=files, headers=headers)

    # Process the Mindee API response as needed
    result = response.json()
    print(json.dumps(result))
    line_items = result['document']['inference']['pages'][0]['prediction']['line_items']
    data = {'description':[],'quantity': [], 'total_amount': [], 'unit_price': []}

    for item in line_items:
        data['description'].append(item['description'])
        data['quantity'].append(item['quantity'])
        data['total_amount'].append(item['total_amount'])
        data['unit_price'].append(item['unit_price'])
    df = pd.DataFrame(data)
    print(df)

    finalsheet,total=find_price_difference(df,"output.csv")
    table_html = finalsheet.to_html(index=False, classes='table table-bordered table-hover')

    return jsonify({'result': {}, 'finalsheet': table_html, 'total_loss': total})


def find_price_difference(df, csv_path):
    df1 = pd.DataFrame(columns=['Item', 'Purchased Price','Retail Rise Price', 'Loss'])
    sum=0

    # Load the master CSV file
    master_df = pd.read_csv(csv_path)

    # Ensure that both dataframes have the 'description' and 'unit_price' columns
    if 'description' not in df.columns or 'unit_price' not in df.columns:
        raise ValueError("DataFrame must have 'description' and 'unit_price' columns.")
    if 'description' not in master_df.columns or 'unit_price' not in master_df.columns:
        raise ValueError("Master DataFrame must have 'description' and 'unit_price' columns.")

    # Create a CountVectorizer to convert descriptions to numerical vectors
    vectorizer = CountVectorizer()

    # Fit and transform descriptions from both dataframes
    df_descriptions = vectorizer.fit_transform(df['description'].astype(str))
    master_descriptions = vectorizer.transform(master_df['description'].astype(str))

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(df_descriptions, master_descriptions)

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        description = row['description']
        quantity = row['quantity']
        total_amount = row['total_amount']
        unit_price_df = row['unit_price']

        # Find the index of the most similar description in the master CSV file
        most_similar_index = similarity_matrix[idx].argmax()

        # Get the unit price from the master CSV file
        unit_price_master = master_df.iloc[most_similar_index]['unit_price']

        # Compare unit prices and calculate the difference
        if unit_price_master < unit_price_df:
            price_difference = unit_price_df - unit_price_master
            data = {
                'Item' : description,
                'Purchased Price': unit_price_df,
                'Retail Rise Price': unit_price_master,
                'Loss': price_difference*quantity

            }
            df1 = pd.concat([df1, pd.DataFrame([data])], ignore_index=True)
            sum+=(price_difference*quantity)
            #print(f"For description '{description}': Unit price difference is {price_difference:.2f}")
    return df1,sum

if __name__ == '__main__':
    app.run(debug=True)
