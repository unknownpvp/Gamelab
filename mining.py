from flask import Flask, request, render_template
from InvIdx import *
from classifiersearch import *
from imagesearch import *

app = Flask(__name__)
gamesearch = TextSearch()
gameclassify = naivebayes()
gameimage = ImageSearch()


@app.before_first_request
def preprocess():
    gamesearch.dataFile()
    gameimage.imageFile()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/result/', methods=['POST'])
def result():
    query = request.form['search']
    search = query.lower().split(" ")
    gamesearch.searchquery(search)
    result = gamesearch.searchresult()
    return render_template('display_result.html', document=result, query=query)


@app.route('/classify/', methods=['POST'])
def classify():
    query = request.form.get('classifier')
    categories = []
    percent = []
    priors = []
    genre = gameclassify.classify(query)
    counter = gameclassify.counts
    priors.append(counter)
    for i in genre:
        categories.append(i)
        percent.append(round((genre[i] * 100), 2))

    return render_template('display_classify.html', data=gameclassify.dataset, query=query,
                           categories=categories, percent=percent)


@app.route('/image/', methods=['POST'])
def images():
    query = request.form['imagesearch']
    terms = query.lower().split(" ")
    gameimage.imagequery(terms)
    result = gameimage.imageresult()
    return render_template("display_image.html", image=result, query=query)


if __name__ == '__main__':
    app.run()
