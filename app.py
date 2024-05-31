import sys
sys.path.insert(0, r"..\getSentimental.py")

import os
import io

import pandas as pd

from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from collections import Counter
from textblob import TextBlob

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns
from wordcloud import WordCloud
import base64

from getSentimental import getAnalysis

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = '../uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def textAnalysis(text):

    analysis = TextBlob(text)
    polarity = analysis.polarity
    subjectivity = analysis.subjectivity

    return polarity, subjectivity


def create_plots_file(text):

    try:
        _, df = getAnalysis().getUpdatedText(text)

    except Exception as e:
        # print(f"Error in getAnalysis: {e}")
        raise

    messages = ""
    for _,i in df.iterrows():
        messages = messages + i['message']
    
    abc = dispalyMoods(df)
    
    polarity, subjectivity = textAnalysis(messages)

    words = messages.split()
    word_freq = Counter(words)
    max_word = word_freq.most_common(1)[0][0]

    mood = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

    plots = {}

    fig, ax = plt.subplots(figsize=(10, 6))
    mood_counts = df.groupby(['sender', 'mood']).size().unstack(fill_value=0)
    mood_counts.plot(kind='bar', stacked=True, color=['skyblue', 'orange', 'green', 'red', 'purple'], ax = ax)
    ax.set_title('Count of Each Mood for Different Senders')
    ax.set_xlabel('Sender')
    ax.set_ylabel('Count of Moods')
    ax.set_xticklabels(mood_counts.index, rotation=90)
    ax.legend(title='Mood')
    ax.grid(linestyle='--')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['word_freq'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['Polarity', 'Subjectivity'], [polarity, subjectivity], color=['blue', 'green'])
    ax.set_title('Sentiment Analysis')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_ylim(-1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['sentiment_analysis'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(messages)    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['word_cloud'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)


    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    heatmap_data = df.groupby(['sender', 'day_of_week']).size().unstack(fill_value=0)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data[days_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", cbar=True, ax = ax)
    ax.set_title('Number of Messages by Sender and Day of the Week')
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Sender')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['mood_distribution'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return plots, polarity, subjectivity, max_word, mood, abc

def create_plots(messages):
        
    polarity, subjectivity = textAnalysis(messages)

    words = messages.split()
    word_freq = Counter(words)
    max_word = word_freq.most_common(1)[0][0]

    mood = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

    plots = {}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(word_freq.keys(), word_freq.values())
    ax.set_title('Word Frequency')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=90)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['word_freq'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['Polarity', 'Subjectivity'], [polarity, subjectivity], color=['blue', 'green'])
    ax.set_title('Sentiment Analysis')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_ylim(-1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['sentiment_analysis'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(messages)    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['word_cloud'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    mood_counts = Counter([mood])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(mood_counts.keys(), mood_counts.values(), color=['red', 'blue', 'green'])
    ax.set_title('Mood Distribution')
    ax.set_xlabel('Mood')
    ax.set_ylabel('Count')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plots['mood_distribution'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return plots, polarity, subjectivity, max_word, mood


def dispalyMoods(df):

    sender = df.groupby("sender")
    max_len = max(len(i) for i, _ in sender)
    abc = "\n"
    for i, j in sender:
        mood_counts = Counter(j.mood)
        mood_str = ", ".join([f"{mood} ({count})" for mood, count in mood_counts.items()])
        abc += f"{i.ljust(max_len)}: \t{mood_str}\n"

    return abc


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files['file']
                lines = file.readlines()
                text = [line.decode('utf-8').strip() for line in lines]
            else:
                text = request.form['text']

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            plots, polarity, subjectivity, max_word, mood, abc = create_plots_file(text)

            return render_template('dashboard.html',
                                   sentiment={'polarity': polarity, 'subjectivity': subjectivity},
                                   max_word=max_word,
                                   mood=mood,
                                   text = abc,
                                   plots=plots)
        except Exception as e:
            flash('This file cannot be handled. Please enter a valid WhatsApp downloaded file.', 'danger')
            return redirect(url_for('upload'))

    return render_template('upload.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        try:
            text = request.form['text']
            plots, polarity, subjectivity, max_word, mood = create_plots(text)

            return render_template('dashboard.html',
                                   sentiment={'polarity': polarity, 'subjectivity': subjectivity},
                                   max_word=max_word,
                                   mood=mood,
                                   text = text,
                                   plots=plots)
        except Exception as e:
            flash('There was an error processing the text. Please try again.', 'danger')
            return redirect(url_for('upload'))

    return render_template('upload.html')


@app.route('/download_report', methods=['POST'])
def download_report():

    text = request.form['text']
    polarity = request.form['polarity']
    subjectivity = request.form['subjectivity']
    max_word = request.form['max_word']
    mood = request.form['mood']
    
    report_content = f"""
    Text Analysis Report
    ========================================================================
    
    Sentiment Analysis:
    - Polarity                 : {polarity}
    - Subjectivity             : {subjectivity}
    - Text Analysis            : {text}
    Word with Maximum Frequency: {max_word}
    Mood                       : {mood}
    """
    
    return send_file(io.BytesIO(report_content.encode('utf-8')),
                     as_attachment=True,
                     download_name='report.txt')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)