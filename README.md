
# Text Generation with LSTM

# Generate Goethe Text with LSTM


```python
import spacy
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger','ner'])
nlp.max_length = 1200000
```

## Read File


```python
def read_file(filepath):    
    with open(filepath) as f:
        str_text = f.read()    
    return str_text
```


```python
goethe_faust = read_file('Goethe_Johann_Wolfgang_Faust_Der_Tragödie_erster_Teil.txt')
```

## Separate tokens and delete irrelevant signs


```python
def separate_punc(doc_text):
    return [token.text for token in nlp(doc_text) if token.text not in '\n\n \n\n\n"-#$%&()--*+,-/:;<=>@[\\]^_`{|}~\t\n ']
```


```python
tokens = separate_punc(goethe_faust)
tokens
```




    ['Faust',
     '.',
     'Der',
     'Tragödie',
     'erster',
     'Teil',
     'Habe',
     'nun',
     'ach',
     '!',
     'Philosophie',
     'Juristerei',
     'und',
     'Medizin',
     'Und',
     'leider',
     'auch',
     'Theologie',
     'Durchaus',
     'studiert',
     'mit',
     'heißem',
     'Bemühn',
     '.',
     'Da',
     'steh',
     "'",
     'ich',
     'nun',
     'ich',
     'armer',
     'Tor',
     'Und',
     'bin',
     'so',
     'klug',
     'als',
     'wie',
     'zuvor',
     '!',
     'Heiße',
     'Magister',
     'heiße',
     'Doktor',
     'gar',
     'Und',
     'ziehe',
     'schon',
     'an',
     'die',
     'zehen',
     'Jahr',
     "'",
     'Herauf',
     'herab',
     'und',
     'quer',
     'und',
     'krumm',
     'Meine',
     'Schüler',
     'an',
     'der',
     'Nase',
     'herum',
     '–',
     'Und',
     'sehe',
     'daß',
     'wir',
     'nichts',
     'wissen',
     'können',
     '!',
     'Das',
     'will',
     'mir',
     'schier',
     'das',
     'Herz',
     'verbrennen',
     '.',
     'Zwar',
     'bin',
     'ich',
     'gescheiter',
     'als',
     'alle',
     'die',
     'Laffen',
     'Doktoren',
     'Magister',
     'Schreiber',
     'und',
     'Pfaffen',
     'Mich',
     'plagen',
     'keine',
     'Skrupel',
     'noch',
     'Zweifel',
     'Fürchte',
     'mich',
     'weder',
     'vor',
     'Hölle',
     'noch',
     'Teufel',
     '–',
     'Dafür',
     'ist',
     'mir',
     'auch',
     'alle',
     'Freud',
     "'",
     'entrissen',
     'Bilde',
     'mir',
     'nicht',
     'ein',
     'was',
     'Rechts',
     'zu',
     'wissen',
     'Bilde',
     'mir',
     'nicht',
     'ein',
     'ich',
     'könnte',
     'was',
     'lehren',
     'Die',
     'Menschen',
     'zu',
     'bessern',
     'und',
     'zu',
     'bekehren',
     '.',
     'Auch',
     'hab',
     "'",
     'ich',
     'weder',
     'Gut',
     'noch',
     'Geld',
     'Noch',
     'Ehr',
     "'",
     'und',
     'Herrlichkeit',
     'der',
     'Welt',
     'Es',
     'möchte',
     'kein',
     'Hund',
     'so',
     'länger',
     'leben',
     '!',
     'Drum',
     'hab',
     "'",
     'ich',
     'mich',
     'der',
     'Magie',
     'ergeben',
     'Ob',
     'mir',
     'durch',
     'Geistes',
     'Kraft',
     'und',
     'Mund',
     'Nicht',
     'manch',
     'Geheimnis',
     'würde',
     'kund',
     'Daß',
     'ich',
     'nicht',
     'mehr',
     'mit',
     'sauerm',
     'Schweiß',
     'Zu',
     'sagen',
     'brauche',
     'was',
     'ich',
     'nicht',
     'weiß',
     'Daß',
     'ich',
     'erkenne',
     'was',
     'die',
     'Welt',
     'I',
     'm',
     'Innersten',
     'zusammenhält',
     'Schau',
     "'",
     'alle',
     'Wirkenskraft',
     'und',
     'Samen',
     'Und',
     'tu',
     "'",
     'nicht',
     'mehr',
     'in',
     'Worten',
     'kramen',
     '.',
     'Zum',
     'letztenmal',
     'auf',
     'meine',
     'Pein',
     'Den',
     'ich',
     'so',
     'manche',
     'Mitternacht',
     'An',
     'diesem',
     'Pult',
     'herangewacht',
     'Dann',
     'über',
     'Büchern',
     'und',
     'Papier',
     "Trübsel'ger",
     'Freund',
     'erschienst',
     'du',
     'mir',
     '!',
     'Ach',
     '!',
     'könnt',
     "'",
     'ich',
     'doch',
     'auf',
     'Bergeshöhn',
     'In',
     'deinem',
     'lieben',
     'Lichte',
     'gehn',
     'Um',
     'Bergeshöhle',
     'mit',
     'Geistern',
     'schweben',
     'Auf',
     'Wiesen',
     'in',
     'deinem',
     'Dämmer',
     'weben',
     'Von',
     'allem',
     'Wissensqualm',
     'entladen',
     'In',
     'deinem',
     'Tau',
     'gesund',
     'mich',
     'baden',
     '!',
     'Weh',
     '!',
     'steck',
     "'",
     'ich',
     'in',
     'dem',
     'Kerker',
     'noch',
     '?',
     'Verfluchtes',
     'dumpfes',
     'Mauerloch',
     'Wo',
     'selbst',
     'das',
     'liebe',
     'Himmelslicht',
     'Trüb',
     'durch',
     'gemalte',
     'Scheiben',
     'bricht',
     '!',
     'Beschränkt',
     'von',
     'diesem',
     'Bücherhauf',
     'Den',
     'Würme',
     'nagen',
     'Staub',
     'bedeckt',
     'Den',
     'bis',
     'ans',
     'hohe',
     'Gewölb',
     "'",
     'hinauf',
     'Ein',
     'angeraucht',
     'Papier',
     'umsteckt',
     'Mit',
     'Gläsern',
     'Büchsen',
     'rings',
     'umstellt',
     'Mit',
     'Instrumenten',
     'vollgepfropft',
     'Urväter',
     'Hausrat',
     'drein',
     'gestopft',
     '–',
     'Das',
     'ist',
     'deine',
     'Welt',
     '!',
     'das',
     'heißt',
     'eine',
     'Welt',
     '!',
     'Und',
     'fragst',
     'du',
     'noch',
     'warum',
     'dein',
     'Herz',
     'Sich',
     'bang',
     'in',
     'deinem',
     'Busen',
     'klemmt',
     '?',
     'Warum',
     'ein',
     'unerklärter',
     'Schmerz',
     'Dir',
     'alle',
     'Lebensregung',
     'hemmt',
     '?',
     'Statt',
     'der',
     'lebendigen',
     'Natur',
     'Da',
     'Gott',
     'die',
     'Menschen',
     'schuf',
     'hinein',
     'Umgibt',
     'in',
     'Rauch',
     'und',
     'Moder',
     'nur',
     'Dich',
     'Tiergeripp',
     "'",
     'und',
     'Totenbein',
     '.',
     'Flieh',
     '!',
     'auf',
     '!',
     'hinaus',
     'ins',
     'weite',
     'Land',
     '!',
     'Und',
     'dies',
     'geheimnisvolle',
     'Buch',
     'Von',
     'Nostradamus',
     "'",
     'eigner',
     'Hand',
     'Ist',
     'dir',
     'es',
     'nicht',
     'Geleit',
     'genug',
     '?',
     'Erkennest',
     'dann',
     'der',
     'Sterne',
     'Lauf',
     'Dann',
     'geht',
     'die',
     'Seelenkraft',
     'dir',
     'auf',
     'Wie',
     'spricht',
     'ein',
     'Geist',
     'zum',
     'andern',
     'Geist',
     '.',
     'Umsonst',
     'daß',
     'trocknes',
     'Sinnen',
     'hier',
     'Die',
     "heil'gen",
     'Zeichen',
     'dir',
     'erklärt',
     'Ihr',
     'schwebt',
     'ihr',
     'Geister',
     'neben',
     'mir',
     'Antwortet',
     'mir',
     'wenn',
     'ihr',
     'mich',
     'hört',
     '!',
     'Ha',
     '!',
     'welche',
     'Wonne',
     'fließt',
     'in',
     'diesem',
     'Blick',
     'Auf',
     'einmal',
     'mir',
     'durch',
     'alle',
     'meine',
     'Sinnen',
     '!',
     'Ich',
     'fühle',
     'junges',
     "heil'ges",
     'Lebensglück',
     'Neuglühend',
     'mir',
     'durch',
     'Nerv',
     "'",
     'und',
     'Adern',
     'rinnen',
     '.',
     'War',
     'es',
     'ein',
     'Gott',
     'der',
     'diese',
     'Zeichen',
     'schrieb',
     'Die',
     'mir',
     'das',
     'innre',
     'Toben',
     'stillen',
     'Das',
     'arme',
     'Herz',
     'mit',
     'Freude',
     'füllen',
     'Und',
     'mit',
     'geheimnisvollem',
     'Trieb',
     'Die',
     'Kräfte',
     'der',
     'Natur',
     'rings',
     'um',
     'mich',
     'her',
     'enthüllen',
     '?',
     'Bin',
     'ich',
     'ein',
     'Gott',
     '?',
     'Mir',
     'wird',
     'so',
     'licht',
     '!',
     'Ich',
     'schau',
     "'",
     'in',
     'diesen',
     'reinen',
     'Zügen',
     'Die',
     'wirkende',
     'Natur',
     'vor',
     'meiner',
     'Seele',
     'liegen',
     '.',
     'Jetzt',
     'erst',
     'erkenn',
     "'",
     'ich',
     'was',
     'der',
     'Weise',
     'spricht',
     '›Die',
     'Geisterwelt',
     'ist',
     'nicht',
     'verschlossen',
     'Dein',
     'Sinn',
     'ist',
     'zu',
     'dein',
     'Herz',
     'ist',
     'tot',
     '!',
     'Auf',
     'bade',
     'Schüler',
     'unverdrossen',
     'Die',
     "ird'sche",
     'Brust',
     'i',
     'm',
     'Morgenrot!‹',
     'Wie',
     'alles',
     'sich',
     'zum',
     'Ganzen',
     'webt',
     'Eins',
     'in',
     'dem',
     'andern',
     'wirkt',
     'und',
     'lebt',
     '!',
     'Wie',
     'Himmelskräfte',
     'auf',
     'und',
     'nieder',
     'steigen',
     'Und',
     'sich',
     'die',
     'goldnen',
     'Eimer',
     'reichen',
     '!',
     'Mit',
     'segenduftenden',
     'Schwingen',
     'Vom',
     'Himmel',
     'durch',
     'die',
     'Erde',
     'dringen',
     'Harmonisch',
     'all',
     'das',
     'All',
     'durchklingen',
     '!',
     'Welch',
     'Schauspiel',
     '!',
     'Aber',
     'ach',
     '!',
     'ein',
     'Schauspiel',
     'nur',
     '!',
     'Wo',
     'fass',
     "'",
     'ich',
     'dich',
     'unendliche',
     'Natur',
     '?',
     'Euch',
     'Brüste',
     'wo',
     '?',
     'Ihr',
     'Quellen',
     'alles',
     'Lebens',
     'An',
     'denen',
     'Himmel',
     'und',
     'Erde',
     'hängt',
     'Dahin',
     'die',
     'welke',
     'Brust',
     'sich',
     'drängt',
     '–',
     'Wie',
     'anders',
     'wirkt',
     'dies',
     'Zeichen',
     'auf',
     'mich',
     'ein',
     '!',
     'Du',
     'Geist',
     'der',
     'Erde',
     'bist',
     'mir',
     'näher',
     'Schon',
     'fühl',
     "'",
     'ich',
     'meine',
     'Kräfte',
     'höher',
     'Schon',
     'glüh',
     "'",
     'ich',
     'wie',
     'von',
     'neuem',
     'Wein',
     'Ich',
     'fühle',
     'Mut',
     'mich',
     'in',
     'die',
     'Welt',
     'zu',
     'wagen',
     'Der',
     'Erde',
     'Weh',
     'der',
     'Erde',
     'Glück',
     'zu',
     'tragen',
     'Mit',
     'Stürmen',
     'mich',
     'herumzuschlagen',
     'Und',
     'in',
     'des',
     'Schiffbruchs',
     'Knirschen',
     'nicht',
     'zu',
     'zagen',
     '.',
     'Es',
     'wölkt',
     'sich',
     'über',
     'mir',
     '–',
     'Der',
     'Mond',
     'verbirgt',
     'sein',
     'Licht',
     '–',
     'Die',
     'Lampe',
     'schwindet',
     '!',
     'Es',
     'dampft',
     '–',
     'Es',
     'zucken',
     'rote',
     'Strahlen',
     'Mir',
     'um',
     'das',
     'Haupt',
     '–',
     'Es',
     'weht',
     'Ein',
     'Schauer',
     'vom',
     'Gewölb',
     "'",
     'herab',
     'Und',
     'faßt',
     'mich',
     'an',
     '!',
     'Ich',
     'fühl',
     "'s",
     'du',
     'schwebst',
     'um',
     'mich',
     'erflehter',
     'Geist',
     '.',
     'Enthülle',
     'dich',
     '!',
     'Ha',
     '!',
     'wie',
     "'s",
     'in',
     'meinem',
     'Herzen',
     'reißt',
     '!',
     'Zu',
     'neuen',
     'Gefühlen',
     'All',
     "'",
     'meine',
     'Sinnen',
     'sich',
     'erwühlen',
     '!',
     'Ich',
     'fühle',
     'ganz',
     'mein',
     'Herz',
     'dir',
     'hingegeben',
     '!',
     'Du',
     'mußt',
     '!',
     'du',
     'mußt',
     '!',
     'und',
     'kostet',
     "'",
     'es',
     'mein',
     'Leben',
     '!',
     'Wer',
     'ruft',
     'mir',
     '?',
     'Schreckliches',
     'Gesicht',
     '!',
     'Du',
     'hast',
     'mich',
     'mächtig',
     'angezogen',
     'An',
     'meiner',
     'Sphäre',
     'lang',
     "'",
     'gesogen',
     'Und',
     'nun',
     '–',
     'Weh',
     '!',
     'ich',
     'ertrag',
     "'",
     'dich',
     'nicht',
     '!',
     'Du',
     'flehst',
     'eratmend',
     'mich',
     'zu',
     'schauen',
     'Meine',
     'Stimme',
     'zu',
     'hören',
     'mein',
     'Antlitz',
     'zu',
     'sehn',
     'Mich',
     'neigt',
     'dein',
     'mächtig',
     'Seelenflehn',
     'Da',
     'bin',
     'ich',
     '!',
     '–',
     'Welch',
     'erbärmlich',
     'Grauen',
     'Faßt',
     'Übermenschen',
     'dich',
     '!',
     'Wo',
     'ist',
     'der',
     'Seele',
     'Ruf',
     '?',
     'Wo',
     'ist',
     'die',
     'Brust',
     'die',
     'eine',
     'Welt',
     'in',
     'sich',
     'erschuf',
     'Und',
     'trug',
     'und',
     'hegte',
     'die',
     'mit',
     'Freudebeben',
     'Wo',
     'bist',
     'du',
     'Faust',
     'des',
     'Stimme',
     'mir',
     'erklang',
     'Der',
     'sich',
     'an',
     'mich',
     'mit',
     'allen',
     'Kräften',
     'drang',
     '?',
     'Bist',
     'du',
     'es',
     'der',
     'von',
     'meinem',
     'Hauch',
     'umwittert',
     'In',
     'allen',
     'Lebenstiefen',
     'zittert',
     'Ein',
     'furchtsam',
     'weggekrümmter',
     'Wurm',
     '?',
     'Soll',
     'ich',
     'dir',
     'Flammenbildung',
     'weichen',
     '?',
     'Ich',
     'bin',
     "'s",
     'bin',
     'Faust',
     'bin',
     'deinesgleichen',
     '!',
     'In',
     'Lebensfluten',
     'i',
     'm',
     'Tatensturm',
     'Wall',
     "'",
     'ich',
     'auf',
     'und',
     'ab',
     'Webe',
     'hin',
     'und',
     'her',
     '!',
     'Geburt',
     'und',
     'Grab',
     'Ein',
     'ewiges',
     'Meer',
     'Ein',
     'wechselnd',
     'Weben',
     'Ein',
     'glühend',
     'Leben',
     'So',
     'schaff',
     "'",
     'ich',
     'am',
     'sausenden',
     'Webstuhl',
     'der',
     'Zeit',
     'Und',
     'wirke',
     'der',
     'Gottheit',
     'lebendiges',
     'Kleid',
     '.',
     'Der',
     'du',
     'die',
     'weite',
     'Welt',
     ...]




```python
# How many tokens does the text have?
len(tokens)
```




    28332



## Create Sequences of Tokens


```python
# how long should sequences be?
train_len = 25+1 

# list of sequences
text_sequences = []

#create sequence list of tokens a,b,c,d,e,f --> a,b,c / b,c,d / c,d,e etc. with train_len = 3
for i in range(train_len, len(tokens)):    
    # Grab train_len amount of characters
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)
```


```python
print(text_sequences[1])
print(text_sequences[2])
```

    ['.', 'Der', 'Tragödie', 'erster', 'Teil', 'Habe', 'nun', 'ach', '!', 'Philosophie', 'Juristerei', 'und', 'Medizin', 'Und', 'leider', 'auch', 'Theologie', 'Durchaus', 'studiert', 'mit', 'heißem', 'Bemühn', '.', 'Da', 'steh', "'"]
    ['Der', 'Tragödie', 'erster', 'Teil', 'Habe', 'nun', 'ach', '!', 'Philosophie', 'Juristerei', 'und', 'Medizin', 'Und', 'leider', 'auch', 'Theologie', 'Durchaus', 'studiert', 'mit', 'heißem', 'Bemühn', '.', 'Da', 'steh', "'", 'ich']


## Keras Tokenization and Convert to Sequence of Numbers (Indices) 


```python
from tensorflow.keras.preprocessing.text import Tokenizer
```


```python
# integer encode sequences of words
tokenizer = Tokenizer(filters='"#$%&()*+/:;<=>@[\\]^_`{|}~\t\n', lower=False, split=' ', char_level=False)
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)
```


```python
sequences[0]
```




    [732,
     2,
     33,
     6045,
     2114,
     494,
     2113,
     72,
     289,
     1,
     6043,
     6041,
     5,
     1305,
     10,
     731,
     54,
     2111,
     6040,
     6039,
     27,
     6038,
     6037,
     2,
     64,
     1306]



## What is behind every index


```python
tokenizer.index_word
```




    {1: '!',
     2: '.',
     3: "'",
     4: 'ich',
     5: 'und',
     6: 'die',
     7: '?',
     8: 'der',
     9: 'nicht',
     10: 'Und',
     11: 'ist',
     12: 'zu',
     13: 'ein',
     14: 'das',
     15: 'Ich',
     16: 'mich',
     17: 'mir',
     18: "'s",
     19: 'du',
     20: 'in',
     21: 'sich',
     22: 'den',
     23: 'so',
     24: 'es',
     25: 'sie',
     26: 'Die',
     27: 'mit',
     28: 'dem',
     29: 'auf',
     30: 'nur',
     31: 'an',
     32: 'Das',
     33: 'Der',
     34: 'Ihr',
     35: 'von',
     36: 'wie',
     37: 'er',
     38: 'was',
     39: 'm',
     40: 'dich',
     41: 'ihr',
     42: 'So',
     43: 'Was',
     44: 'Es',
     45: 'Du',
     46: 'Wie',
     47: 'dir',
     48: 'doch',
     49: 'Ein',
     50: 'i',
     51: 'Euch',
     52: 'Sie',
     53: '–',
     54: 'auch',
     55: 'man',
     56: 'uns',
     57: 'wir',
     58: 'Doch',
     59: 'Mit',
     60: 'Er',
     61: 'noch',
     62: 'wird',
     63: 'wohl',
     64: 'Da',
     65: 'hier',
     66: 'bin',
     67: 'schon',
     68: 'Wenn',
     69: 'In',
     70: 'hat',
     71: 'ihn',
     72: 'nun',
     73: 'war',
     74: 'kann',
     75: 'wenn',
     76: 'des',
     77: 'sein',
     78: 'daß',
     79: 'eine',
     80: 'mein',
     81: 'einen',
     82: 'sind',
     83: 'nach',
     84: 'Als',
     85: 'Daß',
     86: 'Den',
     87: 'soll',
     88: 'für',
     89: 'Welt',
     90: 'als',
     91: 'will',
     92: 'Ist',
     93: 'gar',
     94: 'muß',
     95: 'aus',
     96: 'um',
     97: 'Wir',
     98: 'da',
     99: 'vor',
     100: 'Ach',
     101: 'O',
     102: 'Mann',
     103: 'alles',
     104: 'denn',
     105: 'gleich',
     106: 'nichts',
     107: 'Geist',
     108: 'Nur',
     109: 'einem',
     110: 'zum',
     111: 'am',
     112: 'Mein',
     113: 'recht',
     114: 'alle',
     115: 'bei',
     116: 'gut',
     117: 'Teufel',
     118: 'hab',
     119: 'mehr',
     120: 'gehn',
     121: 'Von',
     122: 'selbst',
     123: 'Gott',
     124: 'Wer',
     125: 'durch',
     126: 'bist',
     127: 'euch',
     128: 'Auf',
     129: 'gern',
     130: 'viel',
     131: 'immer',
     132: 'Nun',
     133: 'wieder',
     134: 'Nicht',
     135: 'meine',
     136: 'Dann',
     137: 'meinem',
     138: 'Herr',
     139: 'Wo',
     140: 'mag',
     141: 'Hier',
     142: 'geht',
     143: 'ganz',
     144: 'Leben',
     145: 'Zeit',
     146: 'dieser',
     147: 'nie',
     148: 'ihm',
     149: 'hast',
     150: 'Kind',
     151: 'fort',
     152: 'Aus',
     153: 'Tag',
     154: 'Mutter',
     155: 'Mir',
     156: 'Herz',
     157: 'Menschen',
     158: 'Ja',
     159: 'Nacht',
     160: 'Herrn',
     161: 'sagen',
     162: 'diesem',
     163: 'ins',
     164: 'erst',
     165: 'allen',
     166: 'steht',
     167: 'haben',
     168: 'kein',
     169: 'I',
     170: 'Freund',
     171: 'meiner',
     172: 'Herzen',
     173: 'Denn',
     174: 'allein',
     175: 'Laß',
     176: 'weiß',
     177: 'An',
     178: 'deine',
     179: 'vom',
     180: 'ja',
     181: 'deinen',
     182: 'wär',
     183: 'sehr',
     184: 'Blut',
     185: 'Mich',
     186: 'Zu',
     187: 'dein',
     188: 'einmal',
     189: 'Brust',
     190: 'sieht',
     191: 'Dem',
     192: 'lange',
     193: 'zur',
     194: 'über',
     195: 'diese',
     196: 'diesen',
     197: 'Schon',
     198: 'lang',
     199: 'Allein',
     200: 'Des',
     201: 'meinen',
     202: 'seh',
     203: 'Kraft',
     204: 'deinem',
     205: 'Weh',
     206: 'her',
     207: 'Man',
     208: 'Haus',
     209: 'sei',
     210: 'eben',
     211: 'seid',
     212: 'keine',
     213: 'Natur',
     214: 'Hand',
     215: 'andern',
     216: 'Himmel',
     217: 'gewiß',
     218: 'davon',
     219: 'kommt',
     220: 'Ende',
     221: 'Komm',
     222: 'läßt',
     223: 'Zum',
     224: 'Busen',
     225: 'dann',
     226: 'sehn',
     227: 'hin',
     228: 'macht',
     229: 'weit',
     230: 'Sind',
     231: 'Durch',
     232: 'Nein',
     233: 'getan',
     234: 'Für',
     235: 'frei',
     236: 'Frau',
     237: 'Sich',
     238: 'Seele',
     239: 'Aber',
     240: 'mußt',
     241: 'hören',
     242: 'Nach',
     243: 'seine',
     244: 'schön',
     245: 'sagt',
     246: 'schönen',
     247: 'laß',
     248: 'geschehn',
     249: 'jedem',
     250: 'gibt',
     251: 'bald',
     252: 'liegt',
     253: 'keinen',
     254: 'Auch',
     255: 'Um',
     256: 'Erde',
     257: 'Soll',
     258: 'Kopf',
     259: 'groß',
     260: 'darf',
     261: 'Hat',
     262: 'Weib',
     263: 'dort',
     264: 'tut',
     265: 'werden',
     266: 'Liebchen',
     267: 'Gretchen',
     268: 'Sinnen',
     269: 'Sinn',
     270: 'nieder',
     271: 'wo',
     272: 'fühl',
     273: 'Kunst',
     274: 'wenig',
     275: 'oft',
     276: 'sonst',
     277: 'tausend',
     278: 'Augen',
     279: 'Vor',
     280: 'ganze',
     281: 'einer',
     282: 'Mädchen',
     283: 'wer',
     284: 'scheint',
     285: 'Will',
     286: 'ganzen',
     287: 'ohne',
     288: 'dran',
     289: 'ach',
     290: 'Hölle',
     291: 'liebe',
     292: 'heißt',
     293: 'hinaus',
     294: 'dies',
     295: 'Freude',
     296: 'Bin',
     297: 'Glück',
     298: 'Tod',
     299: 'Zeiten',
     300: 'kommen',
     301: 'ewig',
     302: 'beim',
     303: 'müssen',
     304: 'Lust',
     305: 'Füßen',
     306: 'heran',
     307: 'jeder',
     308: 'komm',
     309: 'deiner',
     310: 'Schritt',
     311: 'gehen',
     312: 'Tage',
     313: 'seht',
     314: 'ihre',
     315: 'Luft',
     316: 'Wort',
     317: 'machen',
     318: 'oder',
     319: 'Dir',
     320: 'hinein',
     321: 'Blick',
     322: 'aller',
     323: 'Herren',
     324: 'jetzt',
     325: 'Glas',
     326: 'unter',
     327: 'hält',
     328: 'aber',
     329: 'Not',
     330: 'Ei',
     331: 'schöne',
     332: 'geben',
     333: 'heute',
     334: 'Schmuck',
     335: 'Habt',
     336: 'sah',
     337: 'siehst',
     338: 'Kann',
     339: 'Über',
     340: 'habt',
     341: 'etwas',
     342: 'sag',
     343: 'Lieb',
     344: 'Leibe',
     345: 'bis',
     346: 'Warum',
     347: 'genug',
     348: 'Dein',
     349: 'tot',
     350: 'drängt',
     351: 'anders',
     352: 'Wein',
     353: 'tragen',
     354: 'Licht',
     355: 'neuen',
     356: 'nah',
     357: 'kaum',
     358: 'zurück',
     359: 'Sei',
     360: 'kurz',
     361: 'Weg',
     362: 'nennen',
     363: 'Gefühl',
     364: 'hätte',
     365: 'klein',
     366: 'Feuer',
     367: 'Vater',
     368: 'ihren',
     369: 'besten',
     370: 'guten',
     371: 'seiner',
     372: 'ging',
     373: 'seinem',
     374: 'dieses',
     375: 'seinen',
     376: 'habe',
     377: 'leicht',
     378: 'stehn',
     379: 'weh',
     380: 'herein',
     381: 'willst',
     382: 'liebt',
     383: 'ihrem',
     384: 'Ort',
     385: 'Leib',
     386: 'Doktor',
     387: 'Zwar',
     388: 'Gut',
     389: 'möchte',
     390: 'leben',
     391: 'manche',
     392: 'könnt',
     393: 'lieben',
     394: 'allem',
     395: 'Dich',
     396: 'Geister',
     397: 'hört',
     398: 'War',
     399: 'Vom',
     400: 'Lebens',
     401: 'schauen',
     402: 'Meer',
     403: 'möcht',
     404: 'trägt',
     405: 'schwer',
     406: 'andre',
     407: 'los',
     408: 'genießen',
     409: 'hatt',
     410: 'alte',
     411: 'wirst',
     412: 'besser',
     413: 'Lied',
     414: 'Liebe',
     415: 'wollen',
     416: 'sieh',
     417: 'Gesellschaft',
     418: 'dabei',
     419: 'Wird',
     420: 'junge',
     421: 'manchen',
     422: 'Mensch',
     423: 'voll',
     424: 'Bei',
     425: 'eines',
     426: 'ward',
     427: 'Kein',
     428: 'Am',
     429: 'Tier',
     430: 'Pudel',
     431: 'hilft',
     432: 'sollst',
     433: 'gute',
     434: 'drum',
     435: 'dazu',
     436: 'bißchen',
     437: 'heut',
     438: 'Platz',
     439: 'Tor',
     440: 'Jahr',
     441: 'Meine',
     442: 'wissen',
     443: 'würde',
     444: 'Worten',
     445: 'hohe',
     446: 'hinauf',
     447: 'drein',
     448: 'Lauf',
     449: 'welche',
     450: 'arme',
     451: 'Welch',
     452: 'faßt',
     453: 'Gesicht',
     454: 'schaffen',
     455: 'Muß',
     456: 'je',
     457: 'diesmal',
     458: 'Hoffnung',
     459: 'Hab',
     460: 'unsre',
     461: 'Dort',
     462: 'Ruh',
     463: 'stets',
     464: 'lesen',
     465: 'freilich',
     466: 'hätt',
     467: 'alten',
     468: 'werde',
     469: 'letzte',
     470: 'Sonst',
     471: 'Freuden',
     472: 'nehmen',
     473: 'Laßt',
     474: 'Fenster',
     475: 'acht',
     476: 'Seid',
     477: 'Arm',
     478: 'Sohn',
     479: 'niemand',
     480: 'sollt',
     481: 'Glut',
     482: 'damit',
     483: 'Uns',
     484: 'mach',
     485: 'vorbei',
     486: 'Sachen',
     487: 'Kerl',
     488: 'Trank',
     489: 'ihrer',
     490: 'großen',
     491: 'wollt',
     492: 'lieb',
     493: 'glaub',
     494: 'Teil',
     495: 'klug',
     496: 'herum',
     497: 'Drum',
     498: 'Pein',
     499: 'bricht',
     500: 'warum',
     501: 'Schmerz',
     502: 'Buch',
     503: 'Sterne',
     504: 'Wonne',
     505: 'fühle',
     506: 'Eins',
     507: 'Bist',
     508: 'ab',
     509: 'selber',
     510: 'unser',
     511: 'steigt',
     512: 'wahrlich',
     513: 'ersten',
     514: 'Namen',
     515: 'tief',
     516: 'Sinne',
     517: 'jenem',
     518: 'Raum',
     519: 'Wasser',
     520: 'Gift',
     521: 'finden',
     522: 'überall',
     523: 'gewesen',
     524: 'jener',
     525: 'Nichts',
     526: 'Jahre',
     527: 'Christ',
     528: 'singt',
     529: 'Gesang',
     530: 'Genuß',
     531: 'stehen',
     532: 'Bruder',
     533: 'geschwind',
     534: 'verlieren',
     535: 'neue',
     536: 'Frauen',
     537: 'vergebens',
     538: 'regt',
     539: 'nimmt',
     540: 'Menge',
     541: 'Juchhe',
     542: 'Gar',
     543: 'fest',
     544: 'Wissenschaft',
     545: 'Stunde',
     546: 'Berg',
     547: 'schöner',
     548: 'Traum',
     549: 'verloren',
     550: 'Stunden',
     551: 'selig',
     552: 'Glieder',
     553: 'weg',
     554: 'zieht',
     555: 'Schwelle',
     556: 'brennt',
     557: 'weiter',
     558: 'Tür',
     559: 'keiner',
     560: 'Brauch',
     561: 'kannst',
     562: 'bleiben',
     563: 'frisch',
     564: 'fühlen',
     565: 'Einen',
     566: 'Verflucht',
     567: 'Fluch',
     568: 'Hör',
     569: 'tun',
     570: 'ob',
     571: 'guter',
     572: 'große',
     573: 'edlen',
     574: 'grad',
     575: 'führen',
     576: 'Gleich',
     577: 'lassen',
     578: 'Drei',
     579: 'jeden',
     580: 'Worte',
     581: 'früh',
     582: 'Bald',
     583: 'kam',
     584: 'König',
     585: 'lieber',
     586: 'Art',
     587: 'Ding',
     588: 'Engel',
     589: 'halb',
     590: 'herzlich',
     591: 'konnte',
     592: 'Nase',
     593: 'weder',
     594: 'Freud',
     595: 'Geld',
     596: 'Ob',
     597: 'schweben',
     598: 'rings',
     599: 'Land',
     600: 'eigner',
     601: "heil'gen",
     602: 'Zeichen',
     603: 'Ha',
     604: 'liegen',
     605: 'Jetzt',
     606: 'Weise',
     607: 'näher',
     608: 'Mut',
     609: 'Grab',
     610: 'Verzeiht',
     611: 'hör',
     612: 'dringt',
     613: 'gebracht',
     614: 'Jammer',
     615: 'erkennen',
     616: 'rechten',
     617: 'gnug',
     618: 'morgen',
     619: 'findet',
     620: 'fließen',
     621: 'Wahn',
     622: 'fehlt',
     623: 'stand',
     624: 'Riegel',
     625: 'alt',
     626: 'Augenblick',
     627: 'Stelle',
     628: 'Meister',
     629: 'Gunst',
     630: 'Streben',
     631: 'werd',
     632: 'bereit',
     633: 'Dies',
     634: 'keinem',
     635: 'Nachbar',
     636: 'Saft',
     637: 'Gruß',
     638: 'Morgen',
     639: 'welch',
     640: 'Ton',
     641: 'Gewalt',
     642: 'erste',
     643: 'Wunder',
     644: 'Tränen',
     645: 'letzten',
     646: 'süßen',
     647: 'oben',
     648: 'aufs',
     649: 'herauf',
     650: 'nein',
     651: 'Stadt',
     652: 'Heut',
     653: 'zwei',
     654: 'Geschwind',
     655: 'trinkt',
     656: 'Hause',
     657: 'solchen',
     658: 'Hexen',
     659: 'sehen',
     660: 'gewinnen',
     661: 'Sonne',
     662: 'Sieh',
     663: 'bewegt',
     664: 'Band',
     665: 'He',
     666: 'find',
     667: 'jedes',
     668: 'Stein',
     669: 'wert',
     670: 'gegeben',
     671: 'schönes',
     672: 'kennen',
     673: 'herbei',
     674: 'ziehn',
     675: 'Spur',
     676: 'bringen',
     677: 'ruhig',
     678: 'freundlich',
     679: 'heiligen',
     680: 'schätzen',
     681: 'Anfang',
     682: 'schafft',
     683: 'Erst',
     684: 'Wesen',
     685: 'Kannst',
     686: 'Schein',
     687: 'zugrunde',
     688: 'kleine',
     689: 'hoff',
     690: 'Erden',
     691: 'regen',
     692: 'magst',
     693: 'sage',
     694: 'durchs',
     695: 'komme',
     696: 'Alle',
     697: 'sitzt',
     698: 'Herein',
     699: 'langen',
     700: 'jung',
     701: 'Knecht',
     702: 'bequemen',
     703: 'Spiel',
     704: 'Ehre',
     705: 'Stroh',
     706: 'Ihm',
     707: 'Eure',
     708: 'müßt',
     709: 'Freiheit',
     710: 'Zeitvertreib',
     711: 'Ordnung',
     712: 'Zur',
     713: 'halt',
     714: 'rechte',
     715: 'Weiber',
     716: 'Wohin',
     717: 'Bock',
     718: 'genung',
     719: 'Vielleicht',
     720: 'Floh',
     721: 'Volk',
     722: 'Seh',
     723: 'führe',
     724: 'schlecht',
     725: 'Spaß',
     726: 'Schelm',
     727: 'denken',
     728: 'Schatz',
     729: 'bitte',
     730: 'genau',
     731: 'leider',
     732: 'Faust',
     733: 'armer',
     734: 'herab',
     735: 'sehe',
     736: 'plagen',
     737: 'Zweifel',
     738: 'Rechts',
     739: 'Noch',
     740: 'Hund',
     741: 'Geistes',
     742: 'tu',
     743: 'bang',
     744: 'stillen',
     745: 'Trieb',
     746: 'Kräfte',
     747: 'wirkt',
     748: 'all',
     749: 'Schauer',
     750: 'Antlitz',
     751: 'Gottheit',
     752: 'zusammen',
     753: 'Menschheit',
     754: 'Mittel',
     755: 'sterben',
     756: 'Trunk',
     757: 'Grund',
     758: 'bitt',
     759: 'froh',
     760: 'Wahrheit',
     761: 'Sein',
     762: 'büßen',
     763: 'halten',
     764: 'Ins',
     765: 'meiden',
     766: 'Taten',
     767: 'herrliche',
     768: 'Gefühle',
     769: 'Tritt',
     770: 'leichten',
     771: 'hebt',
     772: 'weil',
     773: 'Wald',
     774: 'fasse',
     775: 'dunkeln',
     776: 'engen',
     777: 'vielen',
     778: 'jenen',
     779: 'tönt',
     780: 'Klang',
     781: 'holdes',
     782: 'gehe',
     783: 'Nachbarin',
     784: 'gefällt',
     785: 'Bessers',
     786: 'kehrt',
     787: 'Mag',
     788: 'ließ',
     789: 'begegnen',
     790: 'Lohn',
     791: 'Alles',
     792: 'hervor',
     793: 'Selbst',
     794: 'Juchheisa',
     795: 'Heisa',
     796: 'Dirne',
     797: 'Kreise',
     798: 'wurden',
     799: 'Tropfen',
     800: 'Tagen',
     801: 'Fürwahr',
     802: 'heraus',
     803: 'droben',
     804: 'o',
     805: 'glücklich',
     806: 'Gaben',
     807: 'fragt',
     808: 'fliegen',
     809: 'darum',
     810: 'neues',
     811: 'Flügel',
     812: 'Boden',
     813: 'hinter',
     814: 'strebt',
     815: 'hatte',
     816: 'satt',
     817: 'Zwei',
     818: 'bringt',
     819: 'Feld',
     820: 'Nebel',
     821: 'eng',
     822: 'Stock',
     823: 'tiefe',
     824: 'Gottes',
     825: 'wider',
     826: 'kennt',
     827: 'fängt',
     828: 'ihnen',
     829: 'lernen',
     830: 'hoch',
     831: 'bleibe',
     832: 'Gesellen',
     833: 'beiden',
     834: 'offen',
     835: 'breit',
     836: 'Könnt',
     837: 'sitzen',
     838: 'lachen',
     839: 'Kleinen',
     840: 'bleibt',
     841: 'Male',
     842: 'Eurer',
     843: 'rein',
     844: 'Mal',
     845: 'zarten',
     846: 'singen',
     847: 'vorüber',
     848: 'hinüber',
     849: 'Laube',
     850: 'Gedanken',
     851: 'Weine',
     852: 'sogleich',
     853: 'kommst',
     854: 'Junker',
     855: 'spielen',
     856: 'Ohren',
     857: 'Armen',
     858: 'Trauben',
     859: 'höchsten',
     860: 'Wollen',
     861: 'ruhn',
     862: 'scheiden',
     863: 'gesehn',
     864: 'neu',
     865: 'Schlag',
     866: 'Hast',
     867: 'böser',
     868: 'grade',
     869: 'einzig',
     870: 'Fuß',
     871: 'Beine',
     872: 'Baum',
     873: 'wünschte',
     874: 'behagen',
     875: 'schnell',
     876: 'etwa',
     877: 'herüber',
     878: 'treiben',
     879: 'Nehmt',
     880: 'Damit',
     881: 'übel',
     882: 'geboren',
     883: 'fehlen',
     884: 'trefflich',
     885: 'Blocksberg',
     886: 'Leute',
     887: 'Angst',
     888: 'armen',
     889: 'Reise',
     890: 'Gib',
     891: 'versteht',
     892: 'Gewiß',
     893: 'begehrt',
     894: 'rief',
     895: 'waren',
     896: 'gäb',
     897: 'spüre',
     898: 'Wär',
     899: 'Seht',
     900: 'Fasse',
     901: 'just',
     902: 'gehört',
     903: 'Beim',
     904: 'Dieb',
     905: 'darfst',
     906: 'fast',
     907: 'Au',
     908: 'Leuten',
     909: 'Bösen',
     910: 'stinkt',
     911: 'Genug',
     912: 'Fräulein',
     913: 'sprach',
     914: 'Geht',
     915: 'süße',
     916: 'konnt',
     917: 'Fort',
     918: 'Kästchen',
     919: 'Kirche',
     920: 'Ring',
     921: 'leid',
     922: 'heimlich',
     923: 'armes',
     924: 'Eurem',
     925: 'Lebt',
     926: 'Garten',
     927: 'Felsen',
     928: 'Heinrich',
     929: 'Gegenwart',
     930: 'Kommt',
     931: 'leuchtet',
     932: 'Laub',
     933: 'Rette',
     934: 'Magister',
     935: 'quer',
     936: 'Schüler',
     937: 'können',
     938: 'schier',
     939: 'Pfaffen',
     940: 'könnte',
     941: 'lehren',
     942: 'länger',
     943: 'Mund',
     944: 'Geheimnis',
     945: 'brauche',
     946: 'Schau',
     947: 'Mitternacht',
     948: 'Papier',
     949: 'Geistern',
     950: 'weben',
     951: 'Tau',
     952: 'Verfluchtes',
     953: 'Staub',
     954: 'ans',
     955: 'spricht',
     956: 'schwebt',
     957: 'neben',
     958: 'junges',
     959: 'Adern',
     960: 'schau',
     961: 'webt',
     962: 'goldnen',
     963: 'reichen',
     964: 'denen',
     965: 'hängt',
     966: 'wagen',
     967: 'Stürmen',
     968: 'Mond',
     969: 'Lampe',
     970: 'Haupt',
     971: 'ruft',
     972: 'mächtig',
     973: 'angezogen',
     974: 'Stimme',
     975: 'erbärmlich',
     976: 'Grauen',
     977: 'furchtsam',
     978: 'Ebenbild',
     979: 'klopft',
     980: 'kenn',
     981: 'Fülle',
     982: 'Pfarrer',
     983: 'weiten',
     984: 'fühlt',
     985: 'werdet',
     986: 'andrer',
     987: 'Flammen',
     988: 'Kindern',
     989: 'darnach',
     990: 'Gewinn',
     991: 'eh',
     992: 'Pergament',
     993: 'Ergetzen',
     994: 'gedacht',
     995: 'herrlich',
     996: 'sieben',
     997: 'läuft',
     998: 'Munde',
     999: 'Möcht',
     1000: 'Frage',
     ...}



## Indices and their Corresponding Tokens


```python
for i in sequences[0]:
    print(f'{i} : {tokenizer.index_word[i]}')
```

    732 : Faust
    2 : .
    33 : Der
    6045 : Tragödie
    2114 : erster
    494 : Teil
    2113 : Habe
    72 : nun
    289 : ach
    1 : !
    6043 : Philosophie
    6041 : Juristerei
    5 : und
    1305 : Medizin
    10 : Und
    731 : leider
    54 : auch
    2111 : Theologie
    6040 : Durchaus
    6039 : studiert
    27 : mit
    6038 : heißem
    6037 : Bemühn
    2 : .
    64 : Da
    1306 : steh



```python
tokenizer.word_counts
```




    OrderedDict([('Faust', 105),
                 ('.', 26873),
                 ('Der', 3071),
                 ('Tragödie', 4),
                 ('erster', 31),
                 ('Teil', 162),
                 ('Habe', 33),
                 ('nun', 1386),
                 ('ach', 295),
                 ('!', 27795),
                 ('Philosophie', 11),
                 ('Juristerei', 12),
                 ('und', 10127),
                 ('Medizin', 66),
                 ('Und', 9219),
                 ('leider', 120),
                 ('auch', 2045),
                 ('Theologie', 44),
                 ('Durchaus', 19),
                 ('studiert', 20),
                 ('mit', 3765),
                 ('heißem', 22),
                 ('Bemühn', 23),
                 ('Da', 1663),
                 ('steh', 52),
                 ("'", 13468),
                 ('ich', 11180),
                 ('armer', 104),
                 ('Tor', 182),
                 ('bin', 1534),
                 ('so', 4368),
                 ('klug', 156),
                 ('als', 1144),
                 ('wie', 2860),
                 ('zuvor', 26),
                 ('Heiße', 26),
                 ('Magister', 78),
                 ('heiße', 26),
                 ('Doktor', 208),
                 ('gar', 1118),
                 ('ziehe', 26),
                 ('schon', 1534),
                 ('an', 3172),
                 ('die', 9672),
                 ('zehen', 26),
                 ('Jahr', 182),
                 ('Herauf', 26),
                 ('herab', 104),
                 ('quer', 78),
                 ('krumm', 26),
                 ('Meine', 182),
                 ('Schüler', 78),
                 ('der', 9542),
                 ('Nase', 130),
                 ('herum', 156),
                 ('–', 2054),
                 ('sehe', 104),
                 ('daß', 1274),
                 ('wir', 1898),
                 ('nichts', 962),
                 ('wissen', 182),
                 ('können', 78),
                 ('Das', 3094),
                 ('will', 1144),
                 ('mir', 5309),
                 ('schier', 78),
                 ('das', 5876),
                 ('Herz', 624),
                 ('verbrennen', 26),
                 ('Zwar', 208),
                 ('gescheiter', 52),
                 ('alle', 910),
                 ('Laffen', 26),
                 ('Doktoren', 26),
                 ('Schreiber', 26),
                 ('Pfaffen', 78),
                 ('Mich', 494),
                 ('plagen', 104),
                 ('keine', 416),
                 ('Skrupel', 26),
                 ('noch', 1768),
                 ('Zweifel', 104),
                 ('Fürchte', 26),
                 ('mich', 5382),
                 ('weder', 130),
                 ('vor', 1057),
                 ('Hölle', 286),
                 ('Teufel', 884),
                 ('Dafür', 52),
                 ('ist', 7839),
                 ('Freud', 130),
                 ('entrissen', 26),
                 ('Bilde', 52),
                 ('nicht', 9256),
                 ('ein', 6968),
                 ('was', 2782),
                 ('Rechts', 104),
                 ('zu', 7415),
                 ('könnte', 78),
                 ('lehren', 78),
                 ('Die', 3874),
                 ('Menschen', 624),
                 ('bessern', 26),
                 ('bekehren', 52),
                 ('Auch', 338),
                 ('hab', 858),
                 ('Gut', 208),
                 ('Geld', 130),
                 ('Noch', 104),
                 ('Ehr', 52),
                 ('Herrlichkeit', 52),
                 ('Welt', 1170),
                 ('Es', 2444),
                 ('möchte', 208),
                 ('kein', 546),
                 ('Hund', 104),
                 ('länger', 78),
                 ('leben', 208),
                 ('Drum', 156),
                 ('Magie', 26),
                 ('ergeben', 26),
                 ('Ob', 130),
                 ('durch', 832),
                 ('Geistes', 104),
                 ('Kraft', 442),
                 ('Mund', 78),
                 ('Nicht', 754),
                 ('manch', 52),
                 ('Geheimnis', 78),
                 ('würde', 182),
                 ('kund', 52),
                 ('Daß', 1222),
                 ('mehr', 858),
                 ('sauerm', 26),
                 ('Schweiß', 52),
                 ('Zu', 494),
                 ('sagen', 598),
                 ('brauche', 78),
                 ('weiß', 520),
                 ('erkenne', 26),
                 ('I', 546),
                 ('m', 2730),
                 ('Innersten', 26),
                 ('zusammenhält', 26),
                 ('Schau', 78),
                 ('Wirkenskraft', 26),
                 ('Samen', 26),
                 ('tu', 104),
                 ('in', 5018),
                 ('Worten', 182),
                 ('kramen', 26),
                 ('Zum', 390),
                 ('letztenmal', 26),
                 ('auf', 3224),
                 ('meine', 754),
                 ('Pein', 156),
                 ('Den', 1222),
                 ('manche', 208),
                 ('Mitternacht', 78),
                 ('An', 520),
                 ('diesem', 598),
                 ('Pult', 52),
                 ('herangewacht', 26),
                 ('Dann', 754),
                 ('über', 468),
                 ('Büchern', 52),
                 ('Papier', 78),
                 ("Trübsel'ger", 26),
                 ('Freund', 546),
                 ('erschienst', 26),
                 ('du', 5174),
                 ('Ach', 1040),
                 ('könnt', 208),
                 ('doch', 2236),
                 ('Bergeshöhn', 26),
                 ('In', 1430),
                 ('deinem', 442),
                 ('lieben', 208),
                 ('Lichte', 26),
                 ('gehn', 858),
                 ('Um', 338),
                 ('Bergeshöhle', 26),
                 ('Geistern', 78),
                 ('schweben', 130),
                 ('Auf', 806),
                 ('Wiesen', 52),
                 ('Dämmer', 26),
                 ('weben', 78),
                 ('Von', 858),
                 ('allem', 208),
                 ('Wissensqualm', 26),
                 ('entladen', 26),
                 ('Tau', 78),
                 ('gesund', 52),
                 ('baden', 26),
                 ('Weh', 442),
                 ('steck', 26),
                 ('dem', 3354),
                 ('Kerker', 52),
                 ('?', 9594),
                 ('Verfluchtes', 78),
                 ('dumpfes', 26),
                 ('Mauerloch', 26),
                 ('Wo', 728),
                 ('selbst', 858),
                 ('liebe', 286),
                 ('Himmelslicht', 26),
                 ('Trüb', 26),
                 ('gemalte', 26),
                 ('Scheiben', 26),
                 ('bricht', 156),
                 ('Beschränkt', 26),
                 ('von', 2938),
                 ('Bücherhauf', 26),
                 ('Würme', 26),
                 ('nagen', 26),
                 ('Staub', 78),
                 ('bedeckt', 52),
                 ('bis', 234),
                 ('ans', 78),
                 ('hohe', 182),
                 ('Gewölb', 52),
                 ('hinauf', 182),
                 ('Ein', 2210),
                 ('angeraucht', 52),
                 ('umsteckt', 26),
                 ('Mit', 1846),
                 ('Gläsern', 26),
                 ('Büchsen', 26),
                 ('rings', 130),
                 ('umstellt', 26),
                 ('Instrumenten', 52),
                 ('vollgepfropft', 26),
                 ('Urväter', 26),
                 ('Hausrat', 26),
                 ('drein', 182),
                 ('gestopft', 26),
                 ('deine', 520),
                 ('heißt', 286),
                 ('eine', 1274),
                 ('fragst', 52),
                 ('warum', 156),
                 ('dein', 494),
                 ('Sich', 364),
                 ('bang', 104),
                 ('Busen', 390),
                 ('klemmt', 26),
                 ('Warum', 234),
                 ('unerklärter', 26),
                 ('Schmerz', 156),
                 ('Dir', 260),
                 ('Lebensregung', 26),
                 ('hemmt', 52),
                 ('Statt', 52),
                 ('lebendigen', 26),
                 ('Natur', 416),
                 ('Gott', 858),
                 ('schuf', 52),
                 ('hinein', 260),
                 ('Umgibt', 52),
                 ('Rauch', 52),
                 ('Moder', 26),
                 ('nur', 3198),
                 ('Dich', 208),
                 ('Tiergeripp', 26),
                 ('Totenbein', 26),
                 ('Flieh', 26),
                 ('hinaus', 286),
                 ('ins', 572),
                 ('weite', 52),
                 ('Land', 130),
                 ('dies', 286),
                 ('geheimnisvolle', 26),
                 ('Buch', 156),
                 ('Nostradamus', 26),
                 ('eigner', 130),
                 ('Hand', 416),
                 ('Ist', 1128),
                 ('dir', 2252),
                 ('es', 4316),
                 ('Geleit', 52),
                 ('genug', 234),
                 ('Erkennest', 26),
                 ('dann', 390),
                 ('Sterne', 156),
                 ('Lauf', 182),
                 ('geht', 702),
                 ('Seelenkraft', 26),
                 ('Wie', 2340),
                 ('spricht', 78),
                 ('Geist', 962),
                 ('zum', 936),
                 ('andern', 416),
                 ('Umsonst', 52),
                 ('trocknes', 26),
                 ('Sinnen', 312),
                 ('hier', 1586),
                 ("heil'gen", 130),
                 ('Zeichen', 130),
                 ('erklärt', 26),
                 ('Ihr', 3016),
                 ('schwebt', 78),
                 ('ihr', 2652),
                 ('Geister', 208),
                 ('neben', 78),
                 ('Antwortet', 26),
                 ('wenn', 1326),
                 ('hört', 208),
                 ('Ha', 130),
                 ('welche', 182),
                 ('Wonne', 156),
                 ('fließt', 52),
                 ('Blick', 260),
                 ('einmal', 494),
                 ('Ich', 5538),
                 ('fühle', 156),
                 ('junges', 78),
                 ("heil'ges", 26),
                 ('Lebensglück', 26),
                 ('Neuglühend', 26),
                 ('Nerv', 26),
                 ('Adern', 78),
                 ('rinnen', 26),
                 ('War', 208),
                 ('diese', 468),
                 ('schrieb', 26),
                 ('innre', 26),
                 ('Toben', 26),
                 ('stillen', 104),
                 ('arme', 182),
                 ('Freude', 286),
                 ('füllen', 26),
                 ('geheimnisvollem', 26),
                 ('Trieb', 104),
                 ('Kräfte', 104),
                 ('um', 1092),
                 ('her', 442),
                 ('enthüllen', 26),
                 ('Bin', 286),
                 ('Mir', 644),
                 ('wird', 1690),
                 ('licht', 26),
                 ('schau', 78),
                 ('diesen', 468),
                 ('reinen', 52),
                 ('Zügen', 52),
                 ('wirkende', 26),
                 ('meiner', 546),
                 ('Seele', 364),
                 ('liegen', 130),
                 ('Jetzt', 130),
                 ('erst', 572),
                 ('erkenn', 26),
                 ('Weise', 130),
                 ('›Die', 26),
                 ('Geisterwelt', 26),
                 ('verschlossen', 26),
                 ('Dein', 234),
                 ('Sinn', 312),
                 ('tot', 234),
                 ('bade', 26),
                 ('unverdrossen', 26),
                 ("ird'sche", 26),
                 ('Brust', 494),
                 ('i', 2184),
                 ('Morgenrot!‹', 26),
                 ('alles', 1014),
                 ('sich', 4732),
                 ('Ganzen', 26),
                 ('webt', 78),
                 ('Eins', 156),
                 ('wirkt', 104),
                 ('lebt', 52),
                 ('Himmelskräfte', 26),
                 ('nieder', 312),
                 ('steigen', 52),
                 ('goldnen', 78),
                 ('Eimer', 26),
                 ('reichen', 78),
                 ('segenduftenden', 26),
                 ('Schwingen', 52),
                 ('Vom', 208),
                 ('Himmel', 416),
                 ('Erde', 338),
                 ('dringen', 52),
                 ('Harmonisch', 26),
                 ('all', 104),
                 ('All', 52),
                 ('durchklingen', 26),
                 ('Welch', 182),
                 ('Schauspiel', 52),
                 ('Aber', 364),
                 ('fass', 26),
                 ('dich', 2730),
                 ('unendliche', 26),
                 ('Euch', 2132),
                 ('Brüste', 26),
                 ('wo', 312),
                 ('Quellen', 52),
                 ('Lebens', 208),
                 ('denen', 78),
                 ('hängt', 78),
                 ('Dahin', 26),
                 ('welke', 52),
                 ('drängt', 234),
                 ('anders', 234),
                 ('Du', 2366),
                 ('bist', 832),
                 ('näher', 130),
                 ('Schon', 468),
                 ('fühl', 312),
                 ('höher', 52),
                 ('glüh', 26),
                 ('neuem', 52),
                 ('Wein', 234),
                 ('Mut', 130),
                 ('wagen', 78),
                 ('Glück', 286),
                 ('tragen', 234),
                 ('Stürmen', 78),
                 ('herumzuschlagen', 26),
                 ('des', 1326),
                 ('Schiffbruchs', 26),
                 ('Knirschen', 26),
                 ('zagen', 26),
                 ('wölkt', 26),
                 ('Mond', 78),
                 ('verbirgt', 52),
                 ('sein', 1300),
                 ('Licht', 234),
                 ('Lampe', 78),
                 ('schwindet', 52),
                 ('dampft', 26),
                 ('zucken', 26),
                 ('rote', 26),
                 ('Strahlen', 26),
                 ('Haupt', 78),
                 ('weht', 26),
                 ('Schauer', 104),
                 ('vom', 520),
                 ('faßt', 182),
                 ("'s", 5296),
                 ('schwebst', 26),
                 ('erflehter', 26),
                 ('Enthülle', 26),
                 ('meinem', 754),
                 ('Herzen', 546),
                 ('reißt', 26),
                 ('neuen', 234),
                 ('Gefühlen', 26),
                 ('erwühlen', 26),
                 ('ganz', 702),
                 ('mein', 1274),
                 ('hingegeben', 52),
                 ('mußt', 364),
                 ('kostet', 26),
                 ('Leben', 702),
                 ('Wer', 858),
                 ('ruft', 78),
                 ('Schreckliches', 26),
                 ('Gesicht', 182),
                 ('hast', 676),
                 ('mächtig', 78),
                 ('angezogen', 78),
                 ('Sphäre', 26),
                 ('lang', 468),
                 ('gesogen', 26),
                 ('ertrag', 26),
                 ('flehst', 26),
                 ('eratmend', 26),
                 ('schauen', 208),
                 ('Stimme', 78),
                 ('hören', 364),
                 ('Antlitz', 104),
                 ('sehn', 390),
                 ('neigt', 26),
                 ('Seelenflehn', 26),
                 ('erbärmlich', 78),
                 ('Grauen', 78),
                 ('Faßt', 26),
                 ('Übermenschen', 26),
                 ('Ruf', 26),
                 ('erschuf', 26),
                 ('trug', 52),
                 ('hegte', 26),
                 ('Freudebeben', 26),
                 ('erklang', 26),
                 ('allen', 572),
                 ('Kräften', 52),
                 ('drang', 52),
                 ('Bist', 156),
                 ('Hauch', 26),
                 ('umwittert', 26),
                 ('Lebenstiefen', 26),
                 ('zittert', 52),
                 ('furchtsam', 78),
                 ('weggekrümmter', 26),
                 ('Wurm', 52),
                 ('Soll', 338),
                 ('Flammenbildung', 26),
                 ('weichen', 52),
                 ('deinesgleichen', 52),
                 ('Lebensfluten', 26),
                 ('Tatensturm', 26),
                 ('Wall', 26),
                 ('ab', 156),
                 ('Webe', 26),
                 ('hin', 390),
                 ('Geburt', 26),
                 ('Grab', 130),
                 ('ewiges', 26),
                 ('Meer', 208),
                 ('wechselnd', 26),
                 ('Weben', 52),
                 ('glühend', 26),
                 ('So', 2626),
                 ('schaff', 52),
                 ('am', 936),
                 ('sausenden', 26),
                 ('Webstuhl', 26),
                 ('Zeit', 702),
                 ('wirke', 26),
                 ('Gottheit', 104),
                 ('lebendiges', 26),
                 ('Kleid', 26),
                 ('umschweifst', 26),
                 ('Geschäftiger', 26),
                 ('nah', 234),
                 ('gleichst', 26),
                 ('den', 4446),
                 ('begreifst', 26),
                 ('Wem', 52),
                 ('denn', 1014),
                 ('Ebenbild', 78),
                 ('klopft', 78),
                 ('O', 1040),
                 ('Tod', 286),
                 ('kenn', 78),
                 ('Famulus', 26),
                 ('schönstes', 26),
                 ('zunichte', 26),
                 ('Fülle', 78),
                 ('Gesichte', 26),
                 ('trockne', 26),
                 ('Schleicher', 26),
                 ('stören', 26),
                 ('muß', 1118),
                 ('Verzeiht', 130),
                 ('hör', 130),
                 ('deklamieren', 26),
                 ('last', 26),
                 ('gewiß', 416),
                 ('griechisch', 26),
                 ('Trauerspiel', 26),
                 ('dieser', 702),
                 ('Kunst', 312),
                 ('möcht', 208),
                 ('profitieren', 26),
                 ('Denn', 546),
                 ('heutzutage', 26),
                 ('viel', 780),
                 ('öfters', 26),
                 ('rühmen', 26),
                 ('Komödiant', 52),
                 ('einen', 1274),
                 ('Pfarrer', 78),
                 ('Ja', 624),
                 ('wohl', 1664),
                 ('Zeiten', 286),
                 ('kommen', 286),
                 ('mag', 728),
                 ('man', 2002),
                 ('Museum', 26),
                 ('gebannt', 52),
                 ('sieht', 494),
                 ('kaum', 234),
                 ('Feiertag', 26),
                 ('Kaum', 26),
                 ('Fernglas', 26),
                 ('weiten', 78),
                 ('soll', 1222),
                 ('sie', 4186),
                 ('Überredung', 26),
                 ('leiten', 26),
                 ('Wenn', 1482),
                 ('fühlt', 78),
                 ('werdet', 78),
                 ('erjagen', 26),
                 ('aus', 1118),
                 ('dringt', 130),
                 ('urkräftigem', 26),
                 ('Behagen', 26),
                 ('aller', 260),
                 ('Hörer', 26),
                 ('zwingt', 26),
                 ('Sitzt', 52),
                 ('immer', 780),
                 ('Leimt', 26),
                 ('zusammen', 104),
                 ('Braut', 52),
                 ('Ragout', 26),
                 ('andrer', 78),
                 ('Schmaus', 52),
                 ('blast', 26),
                 ('kümmerlichen', 26),
                 ('Flammen', 78),
                 ('Aus', 650),
                 ('eurem', 52),
                 ('Aschenhäufchen', 26),
                 ('raus', 26),
                 ('Bewundrung', 26),
                 ('Kindern', 78),
                 ('Affen', 26),
                 ('euch', 832),
                 ('darnach', 78),
                 ('Gaumen', 52),
                 ('steht', 572),
                 ('Doch', 1872),
                 ('nie', 702),
                 ('schaffen', 182),
                 ('Allein', 468),
                 ('Vortrag', 26),
                 ('macht', 390),
                 ('Redners', 26),
                 ('weit', 390),
                 ('zurück', 234),
                 ('Such', 26),
                 ('Er', 1794),
                 ('redlichen', 26),
                 ('Gewinn', 78),
                 ('Sei', 234),
                 ('schellenlauter', 26),
                 ('trägt', 208),
                 ('Verstand', 52),
                 ('rechter', 52),
                 ('wenig', 312),
                 ('selber', 156),
                 ('Ernst', 52),
                 ('nötig', 52),
                 ('nachzujagen', 26),
                 ('eure', 52),
                 ('Reden', 26),
                 ('blinkend', 26),
                 ('sind', 1248),
                 ('Menschheit', 104),
                 ('Schnitzel', 26),
                 ('kräuselt', 26),
                 ('Sind', 390),
                 ('unerquicklich', 26),
                 ('Nebelwind', 26),
                 ('herbstlich', 26),
                 ('dürren', 52),
                 ('Blätter', 26),
                 ('säuselt', 26),
                 ('kurz', 234),
                 ('unser', 156),
                 ('bei', 910),
                 ('kritischen', 26),
                 ('Bestreben', 26),
                 ('oft', 312),
                 ('Kopf', 338),
                 ('schwer', 208),
                 ('Mittel', 104),
                 ('erwerben', 26),
                 ('Durch', 390),
                 ('steigt', 156),
                 ('eh', 78),
                 ('halben', 26),
                 ('Weg', 234),
                 ('erreicht', 26),
                 ('Muß', 182),
                 ('sterben', 104),
                 ('Pergament', 78),
                 ("heil'ge", 52),
                 ('Bronnen', 26),
                 ('Woraus', 52),
                 ('Trunk', 104),
                 ('Durst', 52),
                 ('ewig', 286),
                 ('stillt', 52),
                 ('quillt', 52),
                 ('groß', 338),
                 ('Ergetzen', 78),
                 ('versetzen', 26),
                 ('uns', 1950),
                 ('weiser', 52),
                 ('Mann', 1040),
                 ('gedacht', 78),
                 ('zuletzt', 52),
                 ('herrlich', 78),
                 ('gebracht', 130),
                 ('ja', 520),
                 ('Mein', 936),
                 ('Vergangenheit', 26),
                 ('sieben', 78),
                 ('Siegeln', 26),
                 ('Was', 2496),
                 ('Grund', 104),
                 ('Herren', 260),
                 ('bespiegeln', 26),
                 ('wahrlich', 156),
                 ('Jammer', 130),
                 ('Man', 442),
                 ('läuft', 78),
                 ('ersten', 156),
                 ('davon', 416),
                 ('Kehrichtfaß', 26),
                 ('Rumpelkammer', 26),
                 ('höchstens', 26),
                 ('Haupt-', 26),
                 ('Staatsaktion', 26),
                 ('trefflichen', 26),
                 ('pragmatischen', 26),
                 ('Maximen', 26),
                 ('Puppen', 52),
                 ('Munde', 78),
                 ('ziemen', 26),
                 ('Möcht', 78),
                 ('jeglicher', 26),
                 ('erkennen', 130),
                 ('darf', 338),
                 ('Kind', 676),
                 ('beim', 286),
                 ('rechten', 130),
                 ('Namen', 156),
                 ('nennen', 234),
                 ('wenigen', 26),
                 ('erkannt', 26),
                 ('töricht', 52),
                 ('gnug', 130),
                 ('volles', 52),
                 ('wahrten', 26),
                 ('Dem', 494),
                 ('Pöbel', 52),
                 ('Gefühl', 234),
                 ('Schauen', 26),
                 ('offenbarten', 26),
                 ('Hat', 338),
                 ('je', 182),
                 ('gekreuzigt', 26),
                 ('verbrannt', 26),
                 ('bitt', 104),
                 ('tief', 156),
                 ('Nacht', 624),
                 ('Wir', 1092),
                 ('müssen', 286),
                 ('diesmal', 182),
                 ('unterbrechen', 26),
                 ('hätte', 234),
                 ('gern', 806),
                 ('fortgewacht', 26),
                 ('gelehrt', 52),
                 ('besprechen', 52),
                 ('morgen', 130),
                 ('Ostertage', 26),
                 ('Erlaubt', 26),
                 ('andre', 208),
                 ('Frage', 78),
                 ('Eifer', 26),
                 ('Studien', 26),
                 ('beflissen', 26),
                 ('Hoffnung', 182),
                 ('immerfort', 52),
                 ('schalem', 26),
                 ('Zeuge', 26),
                 ('klebt', 52),
                 ("gier'ger", 26),
                 ('nach', 1248),
                 ('Schätzen', 78),
                 ('gräbt', 26),
                 ('froh', 104),
                 ('er', 2808),
                 ('Regenwürmer', 26),
                 ('findet', 130),
                 ('Geisterfülle', 26),
                 ('umgab', 26),
                 ('ertönen', 26),
                 ('für', 1196),
                 ('dank', 26),
                 ('ärmlichsten', 26),
                 ('Erdensöhnen', 26),
                 ('rissest', 26),
                 ('Verzweiflung', 52),
                 ('los', 208),
                 ('Sinne', 156),
                 ('zerstören', 26),
                 ('wollte', 78),
                 ('Erscheinung', 52),
                 ('war', 1352),
                 ('riesengroß', 26),
                 ('recht', 936),
                 ('Zwerg', 26),
                 ('empfinden', 52),
                 ('sollte', 78),
                 ('Ganz', 52),
                 ('gedünkt', 26),
                 ('Spiegel', 52),
                 ("ew'ger", 26),
                 ('Wahrheit', 104),
                 ('Sein', 104),
                 ('genoß', 52),
                 ('Himmelsglanz', 26),
                 ('Klarheit', 26),
                 ('abgestreift', 26),
                 ('Erdensohn', 78),
                 ('Cherub', 26),
                 ('dessen', 78),
                 ('freie', 26),
                 ('fließen', 130),
                 ('schaffend', 26),
                 ('Götterleben', 26),
                 ('genießen', 208),
                 ('ahnungsvoll', 52),
                 ('vermaß', 26),
                 ('büßen', 104),
                 ('Donnerwort', 26),
                 ('hat', 1430),
                 ('hinweggerafft', 52),
                 ('gleichen', 26),
                 ('vermessen', 52),
                 ('Hab', 182),
                 ('anzuziehn', 26),
                 ('besessen', 26),
                 ('hatt', 208),
                 ('halten', 104),
                 ('jenem', 156),
                 ("sel'gen", 26),
                 ('Augenblicke', 52),
                 ('fühlte', 26),
                 ('klein', 234),
                 ('stießest', 52),
                 ('grausam', 26),
                 ('zurücke', 52),
                 ('Ins', 104),
                 ('ungewisse', 26),
                 ('Menschenlos', 26),
                 ('lehret', 78),
                 ('meiden', 104),
                 ('gehorchen', 26),
                 ('Drang', 78),
                 ('unsre', 182),
                 ('Taten', 104),
                 ('gut', 910),
                 ('Leiden', 78),
                 ('Sie', 2120),
                 ('hemmen', 26),
                 ('unsres', 26),
                 ('Gang', 78),
                 ('Herrlichsten', 26),
                 ('empfangen', 78),
                 ('Drängt', 52),
                 ('fremd', 52),
                 ('fremder', 52),
                 ('Stoff', 26),
                 ('Guten', 52),
                 ('gelangen', 78),
                 ('Beßre', 26),
                 ('Trug', 52),
                 ('Wahn', 130),
                 ('gaben', 52),
                 ('herrliche', 104),
                 ('Gefühle', 104),
                 ('Erstarren', 26),
                 ('irdischen', 26),
                 ('Gewühle', 52),
                 ('Phantasie', 78),
                 ('sonst', 312),
                 ('kühnem', 26),
                 ('Flug', 26),
                 ('hoffnungsvoll', 26),
                 ('Ewigen', 26),
                 ('erweitert', 26),
                 ('kleiner', 78),
                 ('Raum', 156),
                 ('Sorge', 26),
                 ('nistet', 26),
                 ('gleich', 1014),
                 ('tiefen', 52),
                 ('Dort', 182),
                 ('wirket', 26),
                 ('geheime', 26),
                 ('Schmerzen', 78),
                 ('Unruhig', 26),
                 ('wiegt', 26),
                 ('störet', 26),
                 ('Lust', 286),
                 ('Ruh', 182),
                 ('deckt', 52),
                 ('stets', 182),
                 ('Masken', 26),
                 ('Haus', 442),
                 ('Hof', 52),
                 ('Weib', 338),
                 ('erscheinen', 26),
                 ('Als', 1248),
                 ('Feuer', 234),
                 ('Wasser', 156),
                 ('Dolch', 52),
                 ('Gift', 156),
                 ('bebst', 26),
                 ('trifft', 52),
                 ('verlierst', 26),
                 ('beweinen', 52),
                 ('Göttern', 52),
                 ('gefühlt', 26),
                 ('Wurme', 26),
                 ('durchwühlt', 26),
                 ('Staube', 52),
                 ('nährend', 26),
                 ('Des', 468),
                 ('Wandrers', 26),
                 ('Tritt', 104),
                 ('vernichtet', 26),
                 ('begräbt', 26),
                 ('Wand', 26),
                 ('hundert', 52),
                 ('Fächern', 26),
                 ('verenget', 26),
                 ('Trödel', 26),
                 ('tausendfachem', 26),
                 ('Tand', 26),
                 ('Mottenwelt', 26),
                 ('dränget', 26),
                 ('Hier', 728),
                 ('finden', 156),
                 ('fehlt', 130),
                 ('vielleicht', 78),
                 ('tausend', 312),
                 ('lesen', 182),
                 ('überall', 156),
                 ('gequält', 52),
                 ('hie', 52),
                 ('da', 1066),
                 ('Glücklicher', 26),
                 ('gewesen', 156),
                 ('grinsest', 26),
                 ('hohler', 26),
                 ('Schädel', 26),
                 ('Hirn', 52),
                 ('meines', 52),
                 ('einst', 52),
                 ('verwirret', 26),
                 ('leichten', 104),
                 ('Tag', 650),
                 ('gesucht', 26),
                 ('Dämmrung', 52),
                 ('jämmerlich', 26),
                 ('geirret', 26),
                 ('Instrumente', 26),
                 ('freilich', 182),
                 ('spottet', 26),
                 ('Rad', 26),
                 ('Kämmen', 26),
                 ('Walz', 26),
                 ('Bügel', 26),
                 ('stand', 130),
                 ('solltet', 78),
                 ('Schlüssel', 52),
                 ('euer', 26),
                 ('Bart', 78),
                 ('kraus', 26),
                 ('hebt', 104),
                 ('Riegel', 130),
                 ('Geheimnisvoll', 26),
                 ('lichten', 26),
                 ('Läßt', 52),
                 ('Schleiers', 26),
                 ('berauben', 26),
                 ('offenbaren', 52),
                 ('zwingst', 26),
                 ('Hebeln', 26),
                 ('Schrauben', 26),
                 ('alt', 130),
                 ('Geräte', 26),
                 ('gebraucht', 26),
                 ('stehst', 78),
                 ('weil', 104),
                 ('Vater', 234),
                 ('brauchte', 52),
                 ('alte', 208),
                 ('Rolle', 26),
                 ('wirst', 208),
                 ('Solang', 26),
                 ('trübe', 78),
                 ('schmauchte', 26),
                 ('Weit', 78),
                 ('besser', 208),
                 ('hätt', 182),
                 ('weniges', 26),
                 ('verpraßt', 26),
                 ('ererbt', 26),
                 ('deinen', 520),
                 ('Vätern', 26),
                 ('Erwirb', 26),
                 ('besitzen', 26),
                 ('nützt', 52),
                 ('schwere', 52),
                 ('Last', 52),
                 ('Nur', 962),
                 ('Augenblick', 130),
                 ...])




```python
vocabulary_size = len(tokenizer.word_counts)
vocabulary_size
```




    6045



## Convert to Numpy Matrix


```python
import numpy as np
```


```python
sequences = np.array(sequences)
```


```python
sequences[0]
```




    array([ 732,    2,   33, 6045, 2114,  494, 2113,   72,  289,    1, 6043,
           6041,    5, 1305,   10,  731,   54, 2111, 6040, 6039,   27, 6038,
           6037,    2,   64, 1306])



# Creating an LSTM based model



```python
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding
```


```python
def create_model(vocabulary_size, seq_len):
    model = Sequential() #instance of sequential model
    model.add(Embedding(vocabulary_size, 25, input_length=seq_len))
    model.add(LSTM(150, return_sequences=True)) #neurons numbers
    model.add(LSTM(150)) #Second Layer
    model.add(Dense(150, activation='relu')) #Add Dense Layer
    model.add(Dense(vocabulary_size, activation='softmax'))    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()    
    return model
```


```python
from tensorflow.keras.utils import to_categorical
```


```python
#Features
X = sequences[:,:-1]
#Label
y = sequences[:,-1] 
```


```python
#Change y into 2 categories, number of classes +1 = because keras holds a 0
y = to_categorical(y, num_classes=vocabulary_size+1)
```


```python
seq_len = X.shape[1] #length is 25
X.shape #sequences and their labels
```




    (28306, 25)



## Train Model


```python
# Summary of Model parameters
model = create_model(vocabulary_size+1, seq_len)
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 25, 25)            151150    
    _________________________________________________________________
    lstm (LSTM)                  (None, 25, 150)           105600    
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 150)               180600    
    _________________________________________________________________
    dense (Dense)                (None, 150)               22650     
    _________________________________________________________________
    dense_1 (Dense)              (None, 6046)              912946    
    =================================================================
    Total params: 1,372,946
    Trainable params: 1,372,946
    Non-trainable params: 0
    _________________________________________________________________



```python
from pickle import dump,load
```


```python
# fit model = train model
model.fit(X, y, batch_size=128, epochs=300,verbose=1)
```

    Train on 28306 samples
    Epoch 1/300
    28306/28306 [==============================] - 11s 391us/sample - loss: 7.3303 - accuracy: 0.0361
    Epoch 2/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 6.8502 - accuracy: 0.0371
    Epoch 3/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 6.7011 - accuracy: 0.0384
    Epoch 4/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 6.5284 - accuracy: 0.0478
    Epoch 5/300
    28306/28306 [==============================] - 9s 332us/sample - loss: 6.3851 - accuracy: 0.0571
    Epoch 6/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 6.2636 - accuracy: 0.0665
    Epoch 7/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 6.1423 - accuracy: 0.0710
    Epoch 8/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 6.0307 - accuracy: 0.0755
    Epoch 9/300
    28306/28306 [==============================] - 9s 314us/sample - loss: 5.9321 - accuracy: 0.0791
    Epoch 10/300
    28306/28306 [==============================] - 6s 215us/sample - loss: 5.8447 - accuracy: 0.0813
    Epoch 11/300
    28306/28306 [==============================] - 6s 222us/sample - loss: 5.7629 - accuracy: 0.0829
    Epoch 12/300
    28306/28306 [==============================] - 6s 218us/sample - loss: 5.6833 - accuracy: 0.0862
    Epoch 13/300
    28306/28306 [==============================] - 6s 225us/sample - loss: 5.6024 - accuracy: 0.0866
    Epoch 14/300
    28306/28306 [==============================] - 6s 211us/sample - loss: 5.5196 - accuracy: 0.0879
    Epoch 15/300
    28306/28306 [==============================] - 6s 220us/sample - loss: 5.4424 - accuracy: 0.0910
    Epoch 16/300
    28306/28306 [==============================] - 6s 218us/sample - loss: 5.3612 - accuracy: 0.0899
    Epoch 17/300
    28306/28306 [==============================] - 6s 221us/sample - loss: 5.2857 - accuracy: 0.0935
    Epoch 18/300
    28306/28306 [==============================] - 6s 221us/sample - loss: 5.2134 - accuracy: 0.0944
    Epoch 19/300
    28306/28306 [==============================] - 6s 223us/sample - loss: 5.1478 - accuracy: 0.0958
    Epoch 20/300
    28306/28306 [==============================] - 6s 212us/sample - loss: 5.0827 - accuracy: 0.0961
    Epoch 21/300
    28306/28306 [==============================] - 6s 219us/sample - loss: 5.0202 - accuracy: 0.0984
    Epoch 22/300
    28306/28306 [==============================] - 6s 221us/sample - loss: 4.9656 - accuracy: 0.0989
    Epoch 23/300
    28306/28306 [==============================] - 7s 232us/sample - loss: 4.9098 - accuracy: 0.1012
    Epoch 24/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 4.8530 - accuracy: 0.1010
    Epoch 25/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 4.7985 - accuracy: 0.1048
    Epoch 26/300
    28306/28306 [==============================] - 9s 331us/sample - loss: 4.7463 - accuracy: 0.1049
    Epoch 27/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 4.6914 - accuracy: 0.1076
    Epoch 28/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 4.6418 - accuracy: 0.1084
    Epoch 29/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 4.5894 - accuracy: 0.1112
    Epoch 30/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 4.5366 - accuracy: 0.1131
    Epoch 31/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 4.4854 - accuracy: 0.1143
    Epoch 32/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 4.4405 - accuracy: 0.1180
    Epoch 33/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 4.3853 - accuracy: 0.1203
    Epoch 34/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 4.3384 - accuracy: 0.1240
    Epoch 35/300
    28306/28306 [==============================] - 9s 331us/sample - loss: 4.2905 - accuracy: 0.1267
    Epoch 36/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 4.2377 - accuracy: 0.1304
    Epoch 37/300
    28306/28306 [==============================] - 9s 321us/sample - loss: 4.1933 - accuracy: 0.1350
    Epoch 38/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 4.1472 - accuracy: 0.1390
    Epoch 39/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 4.1059 - accuracy: 0.1468
    Epoch 40/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 4.0586 - accuracy: 0.1484
    Epoch 41/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 4.0182 - accuracy: 0.1539
    Epoch 42/300
    28306/28306 [==============================] - 9s 331us/sample - loss: 3.9764 - accuracy: 0.1570
    Epoch 43/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 3.9353 - accuracy: 0.1621
    Epoch 44/300
    28306/28306 [==============================] - 9s 321us/sample - loss: 3.8978 - accuracy: 0.1659
    Epoch 45/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 3.8610 - accuracy: 0.1717
    Epoch 46/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 3.8294 - accuracy: 0.1736
    Epoch 47/300
    28306/28306 [==============================] - 9s 322us/sample - loss: 3.7903 - accuracy: 0.1823
    Epoch 48/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 3.7552 - accuracy: 0.1852
    Epoch 49/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 3.7267 - accuracy: 0.1875
    Epoch 50/300
    28306/28306 [==============================] - 9s 320us/sample - loss: 3.6962 - accuracy: 0.1944
    Epoch 51/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 3.6571 - accuracy: 0.2017
    Epoch 52/300
    28306/28306 [==============================] - 9s 322us/sample - loss: 3.6268 - accuracy: 0.2042
    Epoch 53/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 3.5980 - accuracy: 0.2097
    Epoch 54/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 3.5715 - accuracy: 0.2131
    Epoch 55/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 3.5395 - accuracy: 0.2200
    Epoch 56/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 3.5107 - accuracy: 0.2227
    Epoch 57/300
    28306/28306 [==============================] - 9s 322us/sample - loss: 3.4821 - accuracy: 0.2297
    Epoch 58/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 3.4542 - accuracy: 0.2361
    Epoch 59/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 3.4270 - accuracy: 0.2393
    Epoch 60/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 3.4028 - accuracy: 0.2431
    Epoch 61/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 3.3759 - accuracy: 0.2495
    Epoch 62/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 3.3478 - accuracy: 0.2535
    Epoch 63/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 3.3226 - accuracy: 0.2588
    Epoch 64/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 3.3013 - accuracy: 0.2614
    Epoch 65/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 3.2690 - accuracy: 0.2663
    Epoch 66/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 3.2435 - accuracy: 0.2724
    Epoch 67/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 3.2201 - accuracy: 0.2765
    Epoch 68/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 3.1998 - accuracy: 0.2788
    Epoch 69/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 3.1717 - accuracy: 0.2846
    Epoch 70/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 3.1490 - accuracy: 0.2882
    Epoch 71/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 3.1314 - accuracy: 0.2915
    Epoch 72/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 3.1057 - accuracy: 0.2980
    Epoch 73/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 3.0769 - accuracy: 0.3027
    Epoch 74/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 3.0574 - accuracy: 0.3076
    Epoch 75/300
    28306/28306 [==============================] - 9s 333us/sample - loss: 4.1625 - accuracy: 0.2288
    Epoch 76/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 3.9060 - accuracy: 0.2337
    Epoch 77/300
    28306/28306 [==============================] - 9s 320us/sample - loss: 3.5505 - accuracy: 0.2570
    Epoch 78/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 3.0997 - accuracy: 0.2957
    Epoch 79/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 3.0242 - accuracy: 0.3124
    Epoch 80/300
    28306/28306 [==============================] - 9s 331us/sample - loss: 2.9872 - accuracy: 0.3194
    Epoch 81/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 2.9610 - accuracy: 0.3227
    Epoch 82/300
    28306/28306 [==============================] - 9s 321us/sample - loss: 2.9332 - accuracy: 0.3295
    Epoch 83/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 2.9049 - accuracy: 0.3363
    Epoch 84/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 2.8872 - accuracy: 0.3400
    Epoch 85/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 2.8707 - accuracy: 0.3418
    Epoch 86/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 2.8463 - accuracy: 0.3478
    Epoch 87/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 2.8219 - accuracy: 0.3506
    Epoch 88/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 2.8099 - accuracy: 0.3541
    Epoch 89/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 2.7861 - accuracy: 0.3584
    Epoch 90/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 2.7729 - accuracy: 0.3605
    Epoch 91/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 2.7480 - accuracy: 0.3666
    Epoch 92/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 2.7345 - accuracy: 0.3665
    Epoch 93/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 2.7056 - accuracy: 0.3744
    Epoch 94/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 2.6958 - accuracy: 0.3780
    Epoch 95/300
    28306/28306 [==============================] - 9s 321us/sample - loss: 2.6761 - accuracy: 0.3808
    Epoch 96/300
    28306/28306 [==============================] - 9s 334us/sample - loss: 2.6579 - accuracy: 0.3838
    Epoch 97/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 2.6390 - accuracy: 0.3873
    Epoch 98/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 2.6198 - accuracy: 0.3924
    Epoch 99/300
    28306/28306 [==============================] - 9s 332us/sample - loss: 2.6093 - accuracy: 0.3903
    Epoch 100/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 2.5870 - accuracy: 0.3964
    Epoch 101/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 2.5684 - accuracy: 0.3991
    Epoch 102/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 2.5491 - accuracy: 0.4050
    Epoch 103/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 2.5401 - accuracy: 0.4047
    Epoch 104/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 2.5204 - accuracy: 0.4095
    Epoch 105/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 2.4928 - accuracy: 0.4165
    Epoch 106/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 2.4772 - accuracy: 0.4174
    Epoch 107/300
    28306/28306 [==============================] - 9s 332us/sample - loss: 2.4621 - accuracy: 0.4220
    Epoch 108/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 2.4445 - accuracy: 0.4230
    Epoch 109/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 2.4333 - accuracy: 0.4281
    Epoch 110/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 2.4207 - accuracy: 0.4278
    Epoch 111/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 2.4014 - accuracy: 0.4356
    Epoch 112/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 2.3921 - accuracy: 0.4352
    Epoch 113/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 2.3762 - accuracy: 0.4403
    Epoch 114/300
    28306/28306 [==============================] - 9s 332us/sample - loss: 2.3486 - accuracy: 0.4454
    Epoch 115/300
    28306/28306 [==============================] - 9s 322us/sample - loss: 2.3405 - accuracy: 0.4444
    Epoch 116/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 2.3226 - accuracy: 0.4505
    Epoch 117/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 2.3081 - accuracy: 0.4514
    Epoch 118/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 2.2903 - accuracy: 0.4569
    Epoch 119/300
    28306/28306 [==============================] - 9s 333us/sample - loss: 2.2786 - accuracy: 0.4593
    Epoch 120/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 2.2658 - accuracy: 0.4625
    Epoch 121/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 2.2472 - accuracy: 0.4650
    Epoch 122/300
    28306/28306 [==============================] - 9s 322us/sample - loss: 2.2344 - accuracy: 0.4699
    Epoch 123/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 2.2175 - accuracy: 0.4724
    Epoch 124/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 2.2022 - accuracy: 0.4730
    Epoch 125/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 2.2003 - accuracy: 0.4738
    Epoch 126/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 2.1787 - accuracy: 0.4788
    Epoch 127/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 2.1669 - accuracy: 0.4819
    Epoch 128/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 2.1527 - accuracy: 0.4861
    Epoch 129/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 2.1381 - accuracy: 0.4875
    Epoch 130/300
    28306/28306 [==============================] - 9s 332us/sample - loss: 2.1198 - accuracy: 0.4902
    Epoch 131/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 2.1135 - accuracy: 0.4943
    Epoch 132/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 2.0964 - accuracy: 0.4955
    Epoch 133/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 2.0824 - accuracy: 0.4981
    Epoch 134/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 2.0753 - accuracy: 0.5006
    Epoch 135/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 2.0522 - accuracy: 0.5077
    Epoch 136/300
    28306/28306 [==============================] - 7s 253us/sample - loss: 2.0326 - accuracy: 0.5117
    Epoch 137/300
    28306/28306 [==============================] - 6s 214us/sample - loss: 2.0218 - accuracy: 0.5139
    Epoch 138/300
    28306/28306 [==============================] - 6s 206us/sample - loss: 2.0098 - accuracy: 0.5163
    Epoch 139/300
    28306/28306 [==============================] - 6s 211us/sample - loss: 2.0055 - accuracy: 0.5161
    Epoch 140/300
    28306/28306 [==============================] - 6s 223us/sample - loss: 1.9928 - accuracy: 0.5188
    Epoch 141/300
    28306/28306 [==============================] - 6s 207us/sample - loss: 1.9823 - accuracy: 0.5208
    Epoch 142/300
    28306/28306 [==============================] - 6s 220us/sample - loss: 1.9710 - accuracy: 0.5243
    Epoch 143/300
    28306/28306 [==============================] - 6s 214us/sample - loss: 1.9496 - accuracy: 0.5277
    Epoch 144/300
    28306/28306 [==============================] - 6s 226us/sample - loss: 1.9410 - accuracy: 0.5297
    Epoch 145/300
    28306/28306 [==============================] - 6s 216us/sample - loss: 1.9440 - accuracy: 0.5336
    Epoch 146/300
    28306/28306 [==============================] - 6s 222us/sample - loss: 1.9222 - accuracy: 0.5341
    Epoch 147/300
    28306/28306 [==============================] - 6s 216us/sample - loss: 1.9055 - accuracy: 0.5366
    Epoch 148/300
    28306/28306 [==============================] - 6s 215us/sample - loss: 1.8900 - accuracy: 0.5420
    Epoch 149/300
    28306/28306 [==============================] - 6s 216us/sample - loss: 1.8759 - accuracy: 0.5439
    Epoch 150/300
    28306/28306 [==============================] - 6s 207us/sample - loss: 1.8676 - accuracy: 0.5473
    Epoch 151/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 1.8572 - accuracy: 0.5480
    Epoch 152/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 1.8581 - accuracy: 0.5480
    Epoch 153/300
    28306/28306 [==============================] - 9s 321us/sample - loss: 1.8341 - accuracy: 0.5529
    Epoch 154/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 1.8233 - accuracy: 0.5572
    Epoch 155/300
    28306/28306 [==============================] - 9s 322us/sample - loss: 1.8145 - accuracy: 0.5575
    Epoch 156/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.7962 - accuracy: 0.5627
    Epoch 157/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 1.7838 - accuracy: 0.5659
    Epoch 158/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 1.7674 - accuracy: 0.5698
    Epoch 159/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 1.7577 - accuracy: 0.5726
    Epoch 160/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 1.7496 - accuracy: 0.5723
    Epoch 161/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.7438 - accuracy: 0.5731
    Epoch 162/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 1.7249 - accuracy: 0.5780
    Epoch 163/300
    28306/28306 [==============================] - 9s 332us/sample - loss: 1.7099 - accuracy: 0.5827
    Epoch 164/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 1.7085 - accuracy: 0.5834
    Epoch 165/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.6997 - accuracy: 0.5825
    Epoch 166/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.6869 - accuracy: 0.5854
    Epoch 167/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 1.6705 - accuracy: 0.5917
    Epoch 168/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.6551 - accuracy: 0.5949
    Epoch 169/300
    28306/28306 [==============================] - 9s 319us/sample - loss: 1.6569 - accuracy: 0.5916
    Epoch 170/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 1.6295 - accuracy: 0.6013
    Epoch 171/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.6150 - accuracy: 0.6039
    Epoch 172/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 1.6047 - accuracy: 0.6065
    Epoch 173/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 1.6154 - accuracy: 0.6016
    Epoch 174/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.5972 - accuracy: 0.6059
    Epoch 175/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 1.5851 - accuracy: 0.6079
    Epoch 176/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 1.5693 - accuracy: 0.6126
    Epoch 177/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 1.5576 - accuracy: 0.6176
    Epoch 178/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 1.5446 - accuracy: 0.6186
    Epoch 179/300
    28306/28306 [==============================] - 9s 331us/sample - loss: 1.5411 - accuracy: 0.6185
    Epoch 180/300
    28306/28306 [==============================] - 9s 322us/sample - loss: 1.5211 - accuracy: 0.6227
    Epoch 181/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.5092 - accuracy: 0.6286
    Epoch 182/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 1.4975 - accuracy: 0.6307
    Epoch 183/300
    28306/28306 [==============================] - 9s 332us/sample - loss: 1.4878 - accuracy: 0.6334
    Epoch 184/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.4834 - accuracy: 0.6329
    Epoch 185/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 1.4762 - accuracy: 0.6368
    Epoch 186/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.4688 - accuracy: 0.6365
    Epoch 187/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.4514 - accuracy: 0.6401
    Epoch 188/300
    28306/28306 [==============================] - 9s 334us/sample - loss: 1.4345 - accuracy: 0.6462
    Epoch 189/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 1.4186 - accuracy: 0.6510
    Epoch 190/300
    28306/28306 [==============================] - 9s 322us/sample - loss: 1.4212 - accuracy: 0.6476
    Epoch 191/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 1.4179 - accuracy: 0.6483
    Epoch 192/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.4046 - accuracy: 0.6498
    Epoch 193/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.3865 - accuracy: 0.6565
    Epoch 194/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 1.3686 - accuracy: 0.6602
    Epoch 195/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.3580 - accuracy: 0.6622
    Epoch 196/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 1.3375 - accuracy: 0.6685
    Epoch 197/300
    28306/28306 [==============================] - 9s 322us/sample - loss: 1.3354 - accuracy: 0.6699
    Epoch 198/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.3307 - accuracy: 0.6695
    Epoch 199/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 1.3287 - accuracy: 0.6677
    Epoch 200/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 1.3053 - accuracy: 0.6737
    Epoch 201/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.3008 - accuracy: 0.6730
    Epoch 202/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 1.2949 - accuracy: 0.6779
    Epoch 203/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.2730 - accuracy: 0.6812
    Epoch 204/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.2605 - accuracy: 0.6873
    Epoch 205/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 1.2535 - accuracy: 0.6882
    Epoch 206/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 1.2418 - accuracy: 0.6899
    Epoch 207/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.2277 - accuracy: 0.6930
    Epoch 208/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 1.2186 - accuracy: 0.6953
    Epoch 209/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.2119 - accuracy: 0.6971
    Epoch 210/300
    28306/28306 [==============================] - 9s 331us/sample - loss: 1.2137 - accuracy: 0.6955
    Epoch 211/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 1.2037 - accuracy: 0.6967
    Epoch 212/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 1.1853 - accuracy: 0.7032
    Epoch 213/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 1.1849 - accuracy: 0.7027
    Epoch 214/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 1.1678 - accuracy: 0.7072
    Epoch 215/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.1620 - accuracy: 0.7068
    Epoch 216/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.1353 - accuracy: 0.7162
    Epoch 217/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.1197 - accuracy: 0.7193
    Epoch 218/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 1.1152 - accuracy: 0.7195
    Epoch 219/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 1.1229 - accuracy: 0.7191
    Epoch 220/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 1.1008 - accuracy: 0.7250
    Epoch 221/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.0991 - accuracy: 0.7225
    Epoch 222/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.0822 - accuracy: 0.7276
    Epoch 223/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 1.0816 - accuracy: 0.7285
    Epoch 224/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 1.0648 - accuracy: 0.7319
    Epoch 225/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 1.0525 - accuracy: 0.7360
    Epoch 226/300
    28306/28306 [==============================] - 9s 321us/sample - loss: 1.0386 - accuracy: 0.7404
    Epoch 227/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.0396 - accuracy: 0.7399
    Epoch 228/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 1.0205 - accuracy: 0.7431
    Epoch 229/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 1.0111 - accuracy: 0.7464
    Epoch 230/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 1.0208 - accuracy: 0.7445
    Epoch 231/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 1.0112 - accuracy: 0.7455
    Epoch 232/300
    28306/28306 [==============================] - 9s 322us/sample - loss: 0.9933 - accuracy: 0.7508
    Epoch 233/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 0.9681 - accuracy: 0.7576
    Epoch 234/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 0.9677 - accuracy: 0.7587
    Epoch 235/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 0.9535 - accuracy: 0.7621
    Epoch 236/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 0.9493 - accuracy: 0.7617
    Epoch 237/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 0.9354 - accuracy: 0.7645
    Epoch 238/300
    28306/28306 [==============================] - 9s 329us/sample - loss: 0.9327 - accuracy: 0.7667
    Epoch 239/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 0.9189 - accuracy: 0.7695
    Epoch 240/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 0.9197 - accuracy: 0.7675
    Epoch 241/300
    28306/28306 [==============================] - 9s 323us/sample - loss: 0.9199 - accuracy: 0.7675
    Epoch 242/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 0.8978 - accuracy: 0.7751
    Epoch 243/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 0.8881 - accuracy: 0.7758
    Epoch 244/300
    28306/28306 [==============================] - 9s 333us/sample - loss: 0.8782 - accuracy: 0.7784
    Epoch 245/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 0.8713 - accuracy: 0.7824
    Epoch 246/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 0.8711 - accuracy: 0.7806
    Epoch 247/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 0.8621 - accuracy: 0.7814
    Epoch 248/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 0.8506 - accuracy: 0.7853
    Epoch 249/300
    28306/28306 [==============================] - 9s 332us/sample - loss: 0.8300 - accuracy: 0.7928
    Epoch 250/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 0.8175 - accuracy: 0.7946
    Epoch 251/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 0.8114 - accuracy: 0.7992
    Epoch 252/300
    28306/28306 [==============================] - 9s 332us/sample - loss: 0.8004 - accuracy: 0.7996
    Epoch 253/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 0.8032 - accuracy: 0.7980
    Epoch 254/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 0.8022 - accuracy: 0.7983
    Epoch 255/300
    28306/28306 [==============================] - 9s 321us/sample - loss: 0.7918 - accuracy: 0.8010
    Epoch 256/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 0.7724 - accuracy: 0.8052
    Epoch 257/300
    28306/28306 [==============================] - 9s 322us/sample - loss: 0.7540 - accuracy: 0.8124
    Epoch 258/300
    28306/28306 [==============================] - 9s 321us/sample - loss: 0.7514 - accuracy: 0.8134
    Epoch 259/300
    28306/28306 [==============================] - 9s 332us/sample - loss: 0.7440 - accuracy: 0.8164
    Epoch 260/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 0.7647 - accuracy: 0.8078
    Epoch 261/300
    28306/28306 [==============================] - 9s 333us/sample - loss: 0.7484 - accuracy: 0.8111
    Epoch 262/300
    28306/28306 [==============================] - 9s 321us/sample - loss: 0.7240 - accuracy: 0.8185
    Epoch 263/300
    28306/28306 [==============================] - 8s 280us/sample - loss: 0.7010 - accuracy: 0.8261
    Epoch 264/300
    28306/28306 [==============================] - 6s 209us/sample - loss: 0.6764 - accuracy: 0.8369
    Epoch 265/300
    28306/28306 [==============================] - 6s 226us/sample - loss: 0.6931 - accuracy: 0.8279
    Epoch 266/300
    28306/28306 [==============================] - 6s 216us/sample - loss: 0.7025 - accuracy: 0.8225
    Epoch 267/300
    28306/28306 [==============================] - 6s 216us/sample - loss: 0.6877 - accuracy: 0.8285
    Epoch 268/300
    28306/28306 [==============================] - 6s 217us/sample - loss: 0.6901 - accuracy: 0.8243
    Epoch 269/300
    28306/28306 [==============================] - 6s 210us/sample - loss: 0.6794 - accuracy: 0.8281
    Epoch 270/300
    28306/28306 [==============================] - 6s 210us/sample - loss: 0.6693 - accuracy: 0.8315
    Epoch 271/300
    28306/28306 [==============================] - 6s 215us/sample - loss: 0.6572 - accuracy: 0.8357
    Epoch 272/300
    28306/28306 [==============================] - 6s 215us/sample - loss: 0.6313 - accuracy: 0.8453
    Epoch 273/300
    28306/28306 [==============================] - 6s 207us/sample - loss: 0.6312 - accuracy: 0.8445
    Epoch 274/300
    28306/28306 [==============================] - 6s 223us/sample - loss: 0.6168 - accuracy: 0.8471
    Epoch 275/300
    28306/28306 [==============================] - 6s 218us/sample - loss: 0.6105 - accuracy: 0.8487
    Epoch 276/300
    28306/28306 [==============================] - 6s 215us/sample - loss: 0.6067 - accuracy: 0.8520
    Epoch 277/300
    28306/28306 [==============================] - 6s 214us/sample - loss: 0.6084 - accuracy: 0.8511
    Epoch 278/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 0.6233 - accuracy: 0.8429
    Epoch 279/300
    28306/28306 [==============================] - 9s 322us/sample - loss: 0.6133 - accuracy: 0.8469
    Epoch 280/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 0.6000 - accuracy: 0.8516
    Epoch 281/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 0.5807 - accuracy: 0.8566
    Epoch 282/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 0.5515 - accuracy: 0.8661
    Epoch 283/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 0.5350 - accuracy: 0.8707
    Epoch 284/300
    28306/28306 [==============================] - 9s 328us/sample - loss: 0.5347 - accuracy: 0.8702
    Epoch 285/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 0.5471 - accuracy: 0.8650
    Epoch 286/300
    28306/28306 [==============================] - 8s 283us/sample - loss: 0.5637 - accuracy: 0.8584
    Epoch 287/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 0.5324 - accuracy: 0.8695
    Epoch 288/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 0.5352 - accuracy: 0.8697
    Epoch 289/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 0.5296 - accuracy: 0.8699
    Epoch 290/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 0.5094 - accuracy: 0.8747
    Epoch 291/300
    28306/28306 [==============================] - 9s 325us/sample - loss: 0.4877 - accuracy: 0.8832
    Epoch 292/300
    28306/28306 [==============================] - 9s 326us/sample - loss: 0.4815 - accuracy: 0.8843
    Epoch 293/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 0.4850 - accuracy: 0.8844
    Epoch 294/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 0.4975 - accuracy: 0.8776
    Epoch 295/300
    28306/28306 [==============================] - 9s 331us/sample - loss: 0.4875 - accuracy: 0.8800
    Epoch 296/300
    28306/28306 [==============================] - 9s 330us/sample - loss: 0.4992 - accuracy: 0.8772
    Epoch 297/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 0.4806 - accuracy: 0.8822
    Epoch 298/300
    28306/28306 [==============================] - 9s 327us/sample - loss: 0.4591 - accuracy: 0.8880
    Epoch 299/300
    28306/28306 [==============================] - 9s 324us/sample - loss: 0.4392 - accuracy: 0.8966
    Epoch 300/300
    28306/28306 [==============================] - 9s 319us/sample - loss: 0.4360 - accuracy: 0.8958





    <tensorflow.python.keras.callbacks.History at 0x7f9f381e17f0>



## Save Model


```python
# save the model to file
model.save('trainedLSTMModel.h5')
# save the tokenizer
dump(tokenizer, open('trainedLSTMModel', 'wb'))
```

## Generating New Text


```python
from random import randint
from pickle import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
```


```python
def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words): #takes model we trained, tokenizer, sequence length, 
    #seed text to start with (for better results seq_len and seed_text should be the same), number of words to be generated  
  
    # Output
    output_text = []
    
    # Intial Seed Sequence
    input_text = seed_text 
    
    # Create num_gen_words
    for i in range(num_gen_words): #for every generated word. From raw text to seq of numbers
        
        # Take the input text string and encode it to a sequence
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        
        # Pad sequences to our trained rate. If seed text is too short, you need to pad it (fill out or cut it) = match trained rate
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        
        # Predict Class Probabilities for each word
        #going through entire voc, assign a prob to most likely next word 
        #return is the index of word
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        
        # Give the word of the index
        pred_word = tokenizer.index_word[pred_word_ind] 
        
        # take input text, add space and add the predicted word
        input_text += ' ' + pred_word
        
        #append the predicted words
        output_text.append(pred_word)
        
    # return predicted words
    return ' '.join(output_text)
```

## Grab a random seed sequence


```python
import random #grad random text
random.seed(100)
random_pick = random.randint(0,len(text_sequences))
```


```python
#grab random text sequence from text
random_seed_text = text_sequences[random_pick]
```


```python
seed_text = ' '.join(random_seed_text)
```


```python
#generate new text with 200 words 
generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=200)
```




    "Trieb hab ' ich mich der Studien beflissen Zwar weiß ich ist ein gar zu böser Sterne auf ! O müßt Ihr ein moralisch Lied ! Er rief ein schönstes Müßiggang fehlen ! Es klopft . Ach Gott ! mag das ? Wer soll nicht 's . Sie küßt darnach die Maus nachzuahmen . Ach Stunden versteht mich nicht ! Betracht ' ihn recht es ist schön und gut Ungefähr sagt ! bei steht nach dieser Wandel War es der Leuten letzen Und über Mit Als bettelt . Es ist ein Lieben stiller Leben Nur halt ' sich kaum . Gib nur . Was hilft mir selbst ein großer Jammer Bei meinem sich hebt 's der Stelle tot es euch Fliegengott Verderber Lügner heißt . Nun gut ! Nur 's nur wünschen kann . Ich wünschte nicht oft schön ! Küsse wird 's nennen ? Geselle ! ich ein Irrlicht bitte ! Was hinkt der Kerl hier nicht ist ! Es ist nur Spaß Der Takt du Aas Zu deiner Melodei . Aber du beneide schon den Leib Sie nichts meine Pflicht sieht 's wirst du 's hervorgehupft ! Nur eins man mich ? Das ist gerichtet ! Ist den"



## Exploring Generated Sequence


```python
full_text = read_file('Goethe_Johann_Wolfgang_Faust_Der_Tragödie_erster_Teil.txt')
for i,word in enumerate(full_text.split()):
    if word == 'und':
        print(' '.join(full_text.split()[i-20:i+20]))
        print('\n')
```

    
    
    
    bin so klug als wie zuvor! Heiße Magister, heiße Doktor gar, Und ziehe schon an die zehen Jahr' Herauf, herab und quer und krumm Meine Schüler an der Nase herum – Und sehe, daß wir nichts wissen können! Das will
    
    
    klug als wie zuvor! Heiße Magister, heiße Doktor gar, Und ziehe schon an die zehen Jahr' Herauf, herab und quer und krumm Meine Schüler an der Nase herum – Und sehe, daß wir nichts wissen können! Das will mir schier
    
    
    wissen können! Das will mir schier das Herz verbrennen. Zwar bin ich gescheiter als alle die Laffen, Doktoren, Magister, Schreiber und Pfaffen; Mich plagen keine Skrupel noch Zweifel, Fürchte mich weder vor Hölle noch Teufel – Dafür ist mir auch
    
    
    Bilde mir nicht ein, was Rechts zu wissen, Bilde mir nicht ein, ich könnte was lehren, Die Menschen zu bessern und zu bekehren. Auch hab' ich weder Gut noch Geld, Noch Ehr' und Herrlichkeit der Welt; Es möchte kein Hund
    
    
    ich könnte was lehren, Die Menschen zu bessern und zu bekehren. Auch hab' ich weder Gut noch Geld, Noch Ehr' und Herrlichkeit der Welt; Es möchte kein Hund so länger leben! Drum hab' ich mich der Magie ergeben, Ob mir
    
    
    Welt; Es möchte kein Hund so länger leben! Drum hab' ich mich der Magie ergeben, Ob mir durch Geistes Kraft und Mund Nicht manch Geheimnis würde kund; Daß ich nicht mehr mit sauerm Schweiß Zu sagen brauche, was ich nicht
    
    
    Schweiß Zu sagen brauche, was ich nicht weiß; Daß ich erkenne, was die Welt Im Innersten zusammenhält, Schau' alle Wirkenskraft und Samen, Und tu' nicht mehr in Worten kramen. Zum letztenmal auf meine Pein, Den ich so manche Mitternacht An
    
    
    in Worten kramen. Zum letztenmal auf meine Pein, Den ich so manche Mitternacht An diesem Pult herangewacht: Dann über Büchern und Papier, Trübsel'ger Freund, erschienst du mir! Ach! könnt' ich doch auf Bergeshöhn In deinem lieben Lichte gehn, Um Bergeshöhle
    
    
    ein unerklärter Schmerz Dir alle Lebensregung hemmt? Statt der lebendigen Natur, Da Gott die Menschen schuf hinein, Umgibt in Rauch und Moder nur Dich Tiergeripp' und Totenbein. Flieh! auf! hinaus ins weite Land! Und dies geheimnisvolle Buch, Von Nostradamus' eigner
    
    
    Lebensregung hemmt? Statt der lebendigen Natur, Da Gott die Menschen schuf hinein, Umgibt in Rauch und Moder nur Dich Tiergeripp' und Totenbein. Flieh! auf! hinaus ins weite Land! Und dies geheimnisvolle Buch, Von Nostradamus' eigner Hand, Ist dir es nicht
    
    
    fließt in diesem Blick Auf einmal mir durch alle meine Sinnen! Ich fühle junges, heil'ges Lebensglück Neuglühend mir durch Nerv' und Adern rinnen. War es ein Gott, der diese Zeichen schrieb, Die mir das innre Toben stillen, Das arme Herz
    
    
    Auf, bade, Schüler, unverdrossen Die ird'sche Brust im Morgenrot!‹ Wie alles sich zum Ganzen webt, Eins in dem andern wirkt und lebt! Wie Himmelskräfte auf und nieder steigen Und sich die goldnen Eimer reichen! Mit segenduftenden Schwingen Vom Himmel durch
    
    
    ird'sche Brust im Morgenrot!‹ Wie alles sich zum Ganzen webt, Eins in dem andern wirkt und lebt! Wie Himmelskräfte auf und nieder steigen Und sich die goldnen Eimer reichen! Mit segenduftenden Schwingen Vom Himmel durch die Erde dringen, Harmonisch all
    
    
    ach! ein Schauspiel nur! Wo fass' ich dich, unendliche Natur? Euch Brüste, wo? Ihr Quellen alles Lebens, An denen Himmel und Erde hängt, Dahin die welke Brust sich drängt – Wie anders wirkt dies Zeichen auf mich ein! Du, Geist
    
    
    reißt! Zu neuen Gefühlen All' meine Sinnen sich erwühlen! Ich fühle ganz mein Herz dir hingegeben! Du mußt! du mußt! und kostet' es mein Leben! Wer ruft mir? Schreckliches Gesicht! Du hast mich mächtig angezogen, An meiner Sphäre lang' gesogen,
    
    
    Faßt Übermenschen dich! Wo ist der Seele Ruf? Wo ist die Brust, die eine Welt in sich erschuf Und trug und hegte, die mit Freudebeben Wo bist du, Faust, des Stimme mir erklang, Der sich an mich mit allen Kräften
    
    
    weggekrümmter Wurm? Soll ich dir, Flammenbildung, weichen? Ich bin's, bin Faust, bin deinesgleichen! In Lebensfluten, im Tatensturm Wall' ich auf und ab, Webe hin und her! Geburt und Grab, Ein ewiges Meer, Ein wechselnd Weben, Ein glühend Leben, So schaff'
    
    
    dir, Flammenbildung, weichen? Ich bin's, bin Faust, bin deinesgleichen! In Lebensfluten, im Tatensturm Wall' ich auf und ab, Webe hin und her! Geburt und Grab, Ein ewiges Meer, Ein wechselnd Weben, Ein glühend Leben, So schaff' ich am sausenden Webstuhl
    
    
    Ich bin's, bin Faust, bin deinesgleichen! In Lebensfluten, im Tatensturm Wall' ich auf und ab, Webe hin und her! Geburt und Grab, Ein ewiges Meer, Ein wechselnd Weben, Ein glühend Leben, So schaff' ich am sausenden Webstuhl der Zeit Und
    
    
    Leimt zusammen, Braut ein Ragout von andrer Schmaus, Und blast die kümmerlichen Flammen Aus eurem Aschenhäufchen 'raus! Bewundrung von Kindern und Affen, Wenn euch darnach der Gaumen steht – Doch werdet ihr nie Herz zu Herzen schaffen, Wenn es euch
    
    
    es wohl, noch bin ich weit zurück. Such' Er den redlichen Gewinn! Sei Er kein schellenlauter Tor! Es trägt Verstand und rechter Sinn Mit wenig Kunst sich selber vor; Und wenn's euch Ernst ist, was zu sagen, Ist's nötig, Worten
    
    
    Gott! die Kunst ist lang, Und kurz ist unser Leben. Mir wird, bei meinem kritischen Bestreben, Doch oft um Kopf und Busen bang. Wie schwer sind nicht die Mittel zu erwerben, Durch die man zu den Quellen steigt! Und eh'
    
    
    Zeiten sich bespiegeln. Da ist's denn wahrlich oft ein Jammer! Man läuft euch bei dem ersten Blick davon: Ein Kehrichtfaß und eine Rumpelkammer Und höchstens eine Haupt- und Staatsaktion Mit trefflichen pragmatischen Maximen, Wie sie den Puppen wohl im Munde
    
    
    oft ein Jammer! Man läuft euch bei dem ersten Blick davon: Ein Kehrichtfaß und eine Rumpelkammer Und höchstens eine Haupt- und Staatsaktion Mit trefflichen pragmatischen Maximen, Wie sie den Puppen wohl im Munde ziemen! Allein die Welt! des Menschen Herz
    
    
    und Staatsaktion Mit trefflichen pragmatischen Maximen, Wie sie den Puppen wohl im Munde ziemen! Allein die Welt! des Menschen Herz und Geist! Möcht' jeglicher doch was davon erkennen. Ja, was man so erkennen heißt! Wer darf das Kind beim rechten
    
    
    Die töricht gnug ihr volles Herz nicht wahrten, Dem Pöbel ihr Gefühl, ihr Schauen offenbarten, Hat man von je gekreuzigt und verbrannt. Ich bitt' Euch, Freund, es ist tief in der Nacht, Wir müssen's diesmal unterbrechen. Ich hätte gern nur
    
    
    nur immer fortgewacht, Um so gelehrt mit Euch mich zu besprechen. Doch morgen, als am ersten Ostertage, Erlaubt mir ein' und andre Frage. Mit Eifer hab' ich mich der Studien beflissen; Zwar weiß ich viel, doch möcht' ich alles wissen.
    
    
    sollte. Ich, Ebenbild der Gottheit, das sich schon Ganz nah gedünkt dem Spiegel ew'ger Wahrheit, Sein selbst genoß in Himmelsglanz und Klarheit, Und abgestreift den Erdensohn; Ich, mehr als Cherub, dessen freie Kraft Schon durch die Adern der Natur zu
    
    
    so gut als unsre Leiden, Sie hemmen unsres Lebens Gang. Dem Herrlichsten, was auch der Geist empfangen, Drängt immer fremd und fremder Stoff sich an; Wenn wir zum Guten dieser Welt gelangen, Dann heißt das Beßre Trug und Wahn. Die
    
    
    Drängt immer fremd und fremder Stoff sich an; Wenn wir zum Guten dieser Welt gelangen, Dann heißt das Beßre Trug und Wahn. Die uns das Leben gaben, herrliche Gefühle, Erstarren in dem irdischen Gewühle. Wenn Phantasie sich sonst mit kühnem
    
    
    Raum ihr nun genug, Die Sorge nistet gleich im tiefen Herzen, Dort wirket sie geheime Schmerzen, Unruhig wiegt sie sich und störet Lust und Ruh; Sie deckt sich stets mit neuen Masken zu, Sie mag als Haus und Hof, als
    
    
    genug, Die Sorge nistet gleich im tiefen Herzen, Dort wirket sie geheime Schmerzen, Unruhig wiegt sie sich und störet Lust und Ruh; Sie deckt sich stets mit neuen Masken zu, Sie mag als Haus und Hof, als Weib und Kind
    
    
    wiegt sie sich und störet Lust und Ruh; Sie deckt sich stets mit neuen Masken zu, Sie mag als Haus und Hof, als Weib und Kind erscheinen, Als Feuer, Wasser, Dolch und Gift; Du bebst vor allem, was nicht trifft,
    
    
    störet Lust und Ruh; Sie deckt sich stets mit neuen Masken zu, Sie mag als Haus und Hof, als Weib und Kind erscheinen, Als Feuer, Wasser, Dolch und Gift; Du bebst vor allem, was nicht trifft, Und was du nie
    
    
    stets mit neuen Masken zu, Sie mag als Haus und Hof, als Weib und Kind erscheinen, Als Feuer, Wasser, Dolch und Gift; Du bebst vor allem, was nicht trifft, Und was du nie verlierst, das mußt du stets beweinen. Den
    
    
    Dem Wurme gleich' ich, der den Staub durchwühlt, Den, wie er sich im Staube nährend lebt, Des Wandrers Tritt vernichtet und begräbt. Ist es nicht Staub, was diese hohe Wand Aus hundert Fächern mir verenget, Der Trödel, der mit tausendfachem
    
    
    ich finden, was mir fehlt? Soll ich vielleicht in tausend Büchern lesen, Daß überall die Menschen sich gequält, Daß hie und da ein Glücklicher gewesen? – Was grinsest du mir, hohler Schädel, her, Als daß dein Hirn wie meines einst
    
    
    – Was grinsest du mir, hohler Schädel, her, Als daß dein Hirn wie meines einst verwirret Den leichten Tag gesucht und in der Dämmrung schwer, Mit Lust nach Wahrheit, jämmerlich geirret? Ihr Instrumente freilich spottet mein Mit Rad und Kämmen,
    
    
    Tag gesucht und in der Dämmrung schwer, Mit Lust nach Wahrheit, jämmerlich geirret? Ihr Instrumente freilich spottet mein Mit Rad und Kämmen, Walz' und Bügel: Ich stand am Tor, ihr solltet Schlüssel sein; Zwar euer Bart ist kraus, doch hebt
    
    
    in der Dämmrung schwer, Mit Lust nach Wahrheit, jämmerlich geirret? Ihr Instrumente freilich spottet mein Mit Rad und Kämmen, Walz' und Bügel: Ich stand am Tor, ihr solltet Schlüssel sein; Zwar euer Bart ist kraus, doch hebt ihr nicht die
    
    
    des Schleiers nicht berauben, Und was sie deinem Geist nicht offenbaren mag, Das zwingst du ihr nicht ab mit Hebeln und mit Schrauben. Du alt Geräte, das ich nicht gebraucht, Du stehst nur hier, weil dich mein Vater brauchte. Du
    
    
    uns Mondenglanz umweht? Ich grüße dich, du einzige Phiole, Die ich mit Andacht nun herunterhole! In dir verehr' ich Menschenwitz und Kunst. Du Inbegriff der holden Schlummersäfte, Du Auszug aller tödlich feinen Kräfte, Erweise deinem Meister deine Gunst! Ich sehe
    
    
    Ich sehe dich, es wird der Schmerz gelindert, Ich fasse dich, das Streben wird gemindert, Des Geistes Flutstrom ebbet nach und nach. Ins hohe Meer werd' ich hinausgewiesen, Die Spiegelflut erglänzt zu meinen Füßen, Zu neuen Ufern lockt ein neuer
    
    
    neuer Bahn den Äther zu durchdringen, Zu neuen Sphären reiner Tätigkeit. Dies hohe Leben, diese Götterwonne, Du, erst noch Wurm, und die verdienest du? Ja, kehre nur der holden Erdensonne Entschlossen deinen Rücken zu! Vermesse dich, die Pforten aufzureißen, Vor
    
    
    um Grabes Nacht, von Engelslippen klang, Gewißheit einem neuen Bunde? Mit Spezereien Hatten wir ihn gepflegt, Wir seine Treuen Tücher und Binden Reinlich umwanden wir, Ach! und wir finden Christ nicht mehr hier. Christ ist erstanden! Selig der Liebende, Der
    
    
    Gewißheit einem neuen Bunde? Mit Spezereien Hatten wir ihn gepflegt, Wir seine Treuen Tücher und Binden Reinlich umwanden wir, Ach! und wir finden Christ nicht mehr hier. Christ ist erstanden! Selig der Liebende, Der die betrübende, Heilsam' und übende Prüfung
    
    
    umwanden wir, Ach! und wir finden Christ nicht mehr hier. Christ ist erstanden! Selig der Liebende, Der die betrübende, Heilsam' und übende Prüfung bestanden. Was sucht ihr, mächtig und gelind, Ihr Himmelstöne, mich am Staube? Klingt dort umher, wo weiche
    
    
    mehr hier. Christ ist erstanden! Selig der Liebende, Der die betrübende, Heilsam' und übende Prüfung bestanden. Was sucht ihr, mächtig und gelind, Ihr Himmelstöne, mich am Staube? Klingt dort umher, wo weiche Menschen sind. Die Botschaft hör' ich wohl, allein
    
    
    klang so ahnungsvoll des Glockentones Fülle, Und ein Gebet war brünstiger Genuß; Ein unbegreiflich holdes Sehnen Trieb mich, durch Wald und Wiesen hinzugehn, Und unter tausend heißen Tränen Fühlt' ich mir eine Welt entstehn. Dies Lied verkündete der Jugend muntre
    
    
    Was tust denn du? Ich gehe mit den andern. Nach Burgdorf kommt herauf, gewiß dort findet ihr Die schönsten Mädchen und das beste Bier, Und Händel von der ersten Sorte. Du überlustiger Gesell, Juckt dich zum drittenmal das Fell? Ich
    
    
    Gehorchen soll man mehr als immer, Und zahlen mehr als je vorher. Ihr guten Herrn, ihr schönen Frauen, So wohlgeputzt und backenrot, Belieb' es euch, mich anzuschauen, Und seht und mildert meine Not! Laßt hier mich nicht vergebens leiern! Ein
    
    
    als je vorher. Ihr guten Herrn, ihr schönen Frauen, So wohlgeputzt und backenrot, Belieb' es euch, mich anzuschauen, Und seht und mildert meine Not! Laßt hier mich nicht vergebens leiern! Ein Tag, den alle Menschen feiern, Er sei für mich
    
    
    leiern! Ein Tag, den alle Menschen feiern, Er sei für mich ein Erntetag. Nichts Bessers weiß ich mir an Sonn- und Feiertagen Als ein Gespräch von Krieg und Kriegsgeschrei, Wenn hinten, weit, in der Türkei, Die Völker auf einander schlagen.
    
    
    Er sei für mich ein Erntetag. Nichts Bessers weiß ich mir an Sonn- und Feiertagen Als ein Gespräch von Krieg und Kriegsgeschrei, Wenn hinten, weit, in der Türkei, Die Völker auf einander schlagen. Man steht am Fenster, trinkt sein Gläschen
    
    
    aus Und sieht den Fluß hinab die bunten Schiffe gleiten; Dann kehrt man abends froh nach Haus, Und segnet Fried' und Friedenszeiten. Herr Nachbar, ja! so laß ich's auch geschehn, Sie mögen sich die Köpfe spalten, Mag alles durch einander
    
    
    mehreren Verwegnen; Ich seh' mich um, ich such' ihn überall, Allein mir will er nicht begegnen. Burgen mit hohen Mauern und Zinnen, Mädchen mit stolzen Höhnenden Sinnen Möcht' ich gewinnen! Kühn ist das Mühen, Herrlich der Lohn! Lassen wir werben,
    
    
    Lohn! Lassen wir werben, Wie zu der Freude, So zum Verderben. Das ist ein Stürmen! Das ist ein Leben! Mädchen und Burgen Müssen sich geben. Kühn ist das Mühen, Herrlich der Lohn! Und die Soldaten Ziehen davon. Vom Eise befreit
    
    
    Müssen sich geben. Kühn ist das Mühen, Herrlich der Lohn! Und die Soldaten Ziehen davon. Vom Eise befreit sind Strom und Bäche Durch des Frühlings holden, belebenden Blick; Im Tale grünet Hoffnungsglück; Der alte Winter, in seiner Schwäche, Zog sich
    
    
    Ohnmächtige Schauer körnigen Eises In Streifen über die grünende Flur; Aber die Sonne duldet kein Weißes: Überall regt sich Bildung und Streben, Alles will sie mit Farben beleben; Doch an Blumen fehlt's im Revier, Sie nimmt geputzte Menschen dafür. Kehre
    
    
    so gern. Sie feiern die Auferstehung des Herrn, Denn sie sind selber auferstanden, Aus niedriger Häuser dumpfen Gemächern, Aus Handwerks- und Gewerbesbanden, Aus dem Druck von Giebeln und Dächern, Aus der Straßen quetschender Enge, Aus der Kirchen ehrwürdiger Nacht Sieh
    
    
    Herrn, Denn sie sind selber auferstanden, Aus niedriger Häuser dumpfen Gemächern, Aus Handwerks- und Gewerbesbanden, Aus dem Druck von Giebeln und Dächern, Aus der Straßen quetschender Enge, Aus der Kirchen ehrwürdiger Nacht Sieh nur, sieh! wie behend sich die Menge
    
    
    der Straßen quetschender Enge, Aus der Kirchen ehrwürdiger Nacht Sieh nur, sieh! wie behend sich die Menge Durch die Gärten und Felder zerschlägt, Wie der Fluß, in Breit' und Länge, So manchen lustigen Nachen bewegt, Und bis zum Sinken überladen
    
    
    Nacht Sieh nur, sieh! wie behend sich die Menge Durch die Gärten und Felder zerschlägt, Wie der Fluß, in Breit' und Länge, So manchen lustigen Nachen bewegt, Und bis zum Sinken überladen Entfernt sich dieser letzte Kahn. Selbst von des
    
    
    Blinken uns farbige Kleider an. Ich höre schon des Dorfs Getümmel, Hier ist des Volkes wahrer Himmel, Zufrieden jauchzet groß und klein; Hier bin ich Mensch, hier darf ich's sein. Mit Euch, Herr Doktor, zu spazieren, Ist ehrenvoll und ist
    
    
    jauchzet groß und klein; Hier bin ich Mensch, hier darf ich's sein. Mit Euch, Herr Doktor, zu spazieren, Ist ehrenvoll und ist Gewinn; Doch würd' ich nicht allein mich her verlieren, Weil ich ein Feind von allem Rohen bin. Das
    
    
    wie vom bösen Geist getrieben Und nennen's Freude, nennen's Gesang. Der Schäfer putzte sich zum Tanz, Mit bunter Jacke, Band und Kranz, Schmuck war er angezogen. Schon um die Linde war es voll; Und alles tanzte schon wie toll. Juchhe!
    
    
    Heisa! He! Und Hüft' an Ellenbogen. Und tu mir doch nicht so vertraut! Wie mancher hat nicht seine Braut Belogen und betrogen! Er schmeichelte sie doch bei Seit', Und von der Linde scholl es weit: Juchhe! Juchhe! Juchheisa! Heisa! He!
    
    
    betrogen! Er schmeichelte sie doch bei Seit', Und von der Linde scholl es weit: Juchhe! Juchhe! Juchheisa! Heisa! He! Geschrei und Fiedelbogen. Herr Doktor, das ist schön von Euch, Daß Ihr uns heute nicht verschmäht Und unter dieses Volksgedräng', Als
    
    
    ein so Hochgelahrter, geht. So nehmet auch den schönsten Krug, Den wir mit frischem Trunk gefüllt, Ich bring' ihn zu und wünsche laut, Daß er nicht nur den Durst Euch stillt: Die Zahl der Tropfen, die er hegt, Sei Euren
    
    
    stillt: Die Zahl der Tropfen, die er hegt, Sei Euren Tagen zugelegt. Ich nehme den Erquickungstrank, Erwidr' euch allen Heil und Dank. Fürwahr, es ist sehr wohl getan, Daß Ihr am frohen Tag erscheint; Habt Ihr es vormals doch mit
    
    
    Helfer droben. Gesundheit dem bewährten Mann, Daß er noch lange helfen kann! Vor jenem droben steht gebückt, Der helfen lehrt und Hilfe schickt. Welch ein Gefühl mußt du, o großer Mann, Bei der Verehrung dieser Menge haben! O glücklich, wer
    
    
    O glücklich, wer von seinen Gaben Solch einen Vorteil ziehen kann! Der Vater zeigt dich seinem Knaben, Ein jeder fragt und drängt und eilt, Die Fiedel stockt, der Tänzer weilt. Du gehst, in Reihen stehen sie, Die Mützen fliegen in
    
    
    wer von seinen Gaben Solch einen Vorteil ziehen kann! Der Vater zeigt dich seinem Knaben, Ein jeder fragt und drängt und eilt, Die Fiedel stockt, der Tänzer weilt. Du gehst, in Reihen stehen sie, Die Mützen fliegen in die Höh':
    
    
    jenem Stein, Hier wollen wir von unsrer Wandrung rasten. Hier saß ich oft gedankenvoll allein Und quälte mich mit Beten und mit Fasten. An Hoffnung reich, im Glauben fest, Mit Tränen, Seufzen, Händeringen Dacht' ich das Ende jener Pest Vom
    
    
    zu erzwingen. Der Menge Beifall tönt mir nun wie Hohn. O könntest du in meinem Innern lesen, Wie wenig Vater und Sohn Solch eines Ruhmes wert gewesen! Der über die Natur und ihre heil'gen Kreise In Redlichkeit, jedoch auf seine
    
    
    könntest du in meinem Innern lesen, Wie wenig Vater und Sohn Solch eines Ruhmes wert gewesen! Der über die Natur und ihre heil'gen Kreise In Redlichkeit, jedoch auf seine Weise, Mit grillenhafter Mühe sann; Der, in Gesellschaft von Adepten, Sich
    
    
    lobt. Wie könnt Ihr Euch darum betrüben! Tut nicht ein braver Mann genug, Die Kunst, die man ihm übertrug, Gewissenhaft und pünktlich auszuüben? Wenn du, als Jüngling, deinen Vater ehrst, So wirst du gern von ihm empfangen; Wenn du, als
    
    
    Stunde schönes Gut Durch solchen Trübsinn nicht verkümmern! Betrachte, wie in Abendsonneglut Die grünumgebnen Hütten schimmern. Dort eilt sie hin und fördert neues Leben. O daß kein Flügel mich vom Boden hebt, Ihr nach und immer nach zu streben! Ich
    
    
    Hütten schimmern. Dort eilt sie hin und fördert neues Leben. O daß kein Flügel mich vom Boden hebt, Ihr nach und immer nach zu streben! Ich säh' im ewigen Abendstrahl Die stille Welt zu meinen Füßen, Entzündet alle Höhn, beruhigt
    
    
    Göttin endlich wegzusinken; Allein der neue Trieb erwacht, Ich eile fort, ihr ew'ges Licht zu trinken, Vor mir den Tag und hinter mir die Nacht, Den Himmel über mir und unter mir die Wellen. Ein schöner Traum, indessen sie entweicht.
    
    
    eile fort, ihr ew'ges Licht zu trinken, Vor mir den Tag und hinter mir die Nacht, Den Himmel über mir und unter mir die Wellen. Ein schöner Traum, indessen sie entweicht. Ach! zu des Geistes Flügeln wird so leicht Kein
    
    
    des Geistes Flügeln wird so leicht Kein körperlicher Flügel sich gesellen. Doch ist es jedem eingeboren, Daß sein Gefühl hinauf und vorwärts dringt, Wenn über uns, im blauen Raum verloren, Ihr schmetternd Lied die Lerche singt; Wenn über schroffen Fichtenhöhen
    
    
    Ich hatte selbst oft grillenhafte Stunden, Doch solchen Trieb hab' ich noch nie empfunden. Man sieht sich leicht an Wald und Feldern satt; Des Vogels Fittich werd' ich nie beneiden. Wie anders tragen uns die Geistesfreuden Von Buch zu Buch,
    
    
    nie beneiden. Wie anders tragen uns die Geistesfreuden Von Buch zu Buch, von Blatt zu Blatt! Da werden Winternächte hold und schön, Ein selig Leben wärmet alle Glieder, Und ach! entrollst du gar ein würdig Pergamen, So steigt der ganze
    
    
    hebt gewaltsam sich vom Dunst Zu den Gefilden hoher Ahnen. O gibt es Geister in der Luft, Die zwischen Erd' und Himmel herrschend weben, So steiget nieder aus dem goldnen Duft Und führt mich weg, zu neuem, buntem Leben! Ja,
    
    
    schickt, Die Glut auf Glut um deinen Scheitel häufen, So bringt der West den Schwarm, der erst erquickt, Um dich und Feld und Aue zu ersäufen. Sie hören gern, zum Schaden froh gewandt, Gehorchen gern, weil sie uns gern betrügen;
    
    
    Glut auf Glut um deinen Scheitel häufen, So bringt der West den Schwarm, der erst erquickt, Um dich und Feld und Aue zu ersäufen. Sie hören gern, zum Schaden froh gewandt, Gehorchen gern, weil sie uns gern betrügen; Sie stellen
    
    
    die Welt, Die Luft gekühlt, der Nebel fällt! Am Abend schätzt man erst das Haus. – Was stehst du so und blickst erstaunt hinaus? Was kann dich in der Dämmrung so ergreifen? Siehst du den schwarzen Hund durch Saat und
    
    
    so und blickst erstaunt hinaus? Was kann dich in der Dämmrung so ergreifen? Siehst du den schwarzen Hund durch Saat und Stoppel streifen? Ich sah ihn lange schon, nicht wichtig schien er mir. Betracht' ihn recht! für was hältst du
    
    
    auf seine Weise Sich auf der Spur des Herren plagt. Bemerkst du, wie in weitem Schneckenkreise Er um uns her und immer näher jagt? Und irr' ich nicht, so zieht ein Feuerstrudel Auf seinen Pfaden hinterdrein. Ich sehe nichts als
    
    
    sein. Mir scheint es, daß er magisch leise Schlingen Zu künft'gem Band um unsre Füße zieht. Ich seh' ihn ungewiß und furchtsam uns umspringen, Weil er, statt seines Herrn, zwei Unbekannte sieht. Der Kreis wird eng, schon ist er nah!
    
    
    Weil er, statt seines Herrn, zwei Unbekannte sieht. Der Kreis wird eng, schon ist er nah! Du siehst! ein Hund, und kein Gespenst ist da. Er knurrt und zweifelt, legt sich auf den Bauch. Er wedelt. Alles Hundebrauch. Geselle dich
    
    
    sieht. Der Kreis wird eng, schon ist er nah! Du siehst! ein Hund, und kein Gespenst ist da. Er knurrt und zweifelt, legt sich auf den Bauch. Er wedelt. Alles Hundebrauch. Geselle dich zu uns! Komm hier! Es ist ein
    
    
    es bringen, Nach deinem Stock ins Wasser springen. Du hast wohl recht, ich finde nicht die Spur Von einem Geist, und alles ist Dressur. Dem Hunde, wenn er gut gezogen, Wird selbst ein weiser Mann gewogen. Ja, deine Gunst verdient
    
    
    ist Dressur. Dem Hunde, wenn er gut gezogen, Wird selbst ein weiser Mann gewogen. Ja, deine Gunst verdient er ganz und gar, Er, der Studenten trefflicher Skolar. Verlassen hab' ich Feld und Auen, Die eine tiefe Nacht bedeckt, Mit ahnungsvollem,
    
    
    weiser Mann gewogen. Ja, deine Gunst verdient er ganz und gar, Er, der Studenten trefflicher Skolar. Verlassen hab' ich Feld und Auen, Die eine tiefe Nacht bedeckt, Mit ahnungsvollem, heil'gem Grauen In uns die beßre Seele weckt. Mit jedem ungestümen
    
    
    jedem ungestümen Tun; Es reget sich die Menschenliebe, Die Liebe Gottes regt sich nun. Sei ruhig, Pudel! renne nicht hin und wider! An der Schwelle was schnoperst du hier? Lege dich hinter den Ofen nieder, Mein bestes Kissen geb' ich
    
    
    dich hinter den Ofen nieder, Mein bestes Kissen geb' ich dir. Wie du draußen auf dem bergigen Wege Durch Rennen und Springen ergetzt uns hast, So nimm nun auch von mir die Pflege, Als ein willkommner stiller Gast. Ach, wenn
    
    
    tierische Laut nicht passen. Wir sind gewohnt, daß die Menschen verhöhnen, Was sie nicht verstehn, Daß sie vor dem Guten und Schönen, Das ihnen oft beschwerlich ist, murren; Will es der Hund, wie sie, beknurren? Aber ach! schon fühl' ich,
    
    
    Erfahrung. Doch dieser Mangel läßt sich ersetzen: Wir lernen das Überirdische schätzen, Wir sehnen uns nach Offenbarung, Die nirgends würd'ger und schöner brennt Mich drängt's, den Grundtext aufzuschlagen, Mit redlichem Gefühl einmal Das heilige Original In mein geliebtes Deutsch zu
    
    
    der Sinn. Bedenke wohl die erste Zeile, Daß deine Feder sich nicht übereile! Ist es der Sinn, der alles wirkt und schafft? Es sollte stehn: Im Anfang war die Kraft! Doch, auch indem ich dieses niederschreibe, Schon warnt mich was,
    
    
    Lauf. Aber was muß ich sehen! Kann das natürlich geschehen? Ist es Schatten? ist's Wirklichkeit? Wie wird mein Pudel lang und breit! Er hebt sich mit Gewalt, Das ist nicht eines Hundes Gestalt! Welch ein Gespenst bracht' ich ins Haus!
    
    
    folg' ihm keiner! Wie im Eisen der Fuchs, Zagt ein alter Höllenluchs. Aber gebt acht! Schwebet hin, schwebet wider, Auf und nieder, Und er hat sich losgemacht. Könnt ihr ihm nützen, Laßt ihn nicht sitzen! Denn er tat uns allen
    
    
    Geister. Verschwind in Flammen, Salamander! Rauschend fließe zusammen, Undene! Leucht in Meteoren-Schöne, Sylphe! Bring häusliche Hilfe, Incubus! Incubus! Tritt hervor und mache den Schluß. Steckt in dem Tiere. Es liegt ganz ruhig und grinst mich an; Ich hab' ihm noch
    
    
    Sylphe! Bring häusliche Hilfe, Incubus! Incubus! Tritt hervor und mache den Schluß. Steckt in dem Tiere. Es liegt ganz ruhig und grinst mich an; Ich hab' ihm noch nicht weh getan. Du sollst mich hören Stärker beschwören. Bist du Geselle
    
    
    Fliegengott, Verderber, Lügner heißt. Nun gut, wer bist du denn? Ein Teil von jener Kraft, Die stets das Böse will und stets das Gute schafft. Was ist mit diesem Rätselwort gemeint? Ich bin der Geist, der stets verneint! Und das
    
    
    So ist denn alles, was ihr Sünde, Zerstörung, kurz das Böse nennt, Mein eigentliches Element. Du nennst dich einen Teil, und stehst doch ganz vor mir? Bescheidne Wahrheit sprech' ich dir. Wenn sich der Mensch, die kleine Narrenwelt, Gewöhnlich für
    
    
    als ich schon unternommen, Ich wußte nicht ihr beizukommen, Mit Wellen, Stürmen, Schütteln, Brand – Geruhig bleibt am Ende Meer und Land! Und dem verdammten Zeug, der Tier- und Menschenbrut, Dem ist nun gar nichts anzuhaben: Wie viele hab' ich
    
    
    beizukommen, Mit Wellen, Stürmen, Schütteln, Brand – Geruhig bleibt am Ende Meer und Land! Und dem verdammten Zeug, der Tier- und Menschenbrut, Dem ist nun gar nichts anzuhaben: Wie viele hab' ich schon begraben! Und immer zirkuliert ein neues, frisches
    
    
    Der Teufel kann nicht aus dem Haus. Doch warum gehst du nicht durchs Fenster? 's ist ein Gesetz der Teufel und Gespenster: Wo sie hereingeschlüpft, da müssen sie hinaus. Das erste steht uns frei, beim zweiten sind wir Knechte. Die
    
    
    nichts abgezwackt. Doch das ist nicht so kurz zu fassen, Und wir besprechen das zunächst; Doch jetzo bitt' ich hoch und höchst, Für dieses Mal mich zu entlassen. So bleibe doch noch einen Augenblick, Um mir erst gute Mär zu
    
    
    ihn in ein Meer des Wahns; Bedarf ich eines Rattenzahns. Nicht lange brauch' ich zu beschwören, Schon raschelt eine hier und wird sogleich mich hören. Der Herr der Ratten und der Mäuse, Der Fliegen, Frösche, Wanzen, Läuse Befiehlt dir, dich
    
    
    Rattenzahns. Nicht lange brauch' ich zu beschwören, Schon raschelt eine hier und wird sogleich mich hören. Der Herr der Ratten und der Mäuse, Der Fliegen, Frösche, Wanzen, Läuse Befiehlt dir, dich hervorzuwagen Und diese Schwelle zu benagen, Sowie er sie
    
    
    Das Mäntelchen von starrer Seide, Die Hahnenfeder auf dem Hut, Mit einem langen spitzen Degen, Und rate nun dir, kurz und gut, Dergleichen gleichfalls anzulegen; Erfahrest, was das Leben sei. In jedem Kleide werd' ich wohl die Pein Des engen
    
    
    zog, Den Rest von kindlichem Gefühle Mit Anklang froher Zeit betrog, So fluch' ich allem, was die Seele Mit Lock- und Gaukelwerk umspannt, Und sie in diese Trauerhöhle Mit Blend- und Schmeichelkräften bannt! Verflucht voraus die hohe Meinung, Womit der
    
    
    betrog, So fluch' ich allem, was die Seele Mit Lock- und Gaukelwerk umspannt, Und sie in diese Trauerhöhle Mit Blend- und Schmeichelkräften bannt! Verflucht voraus die hohe Meinung, Womit der Geist sich selbst umfängt! Verflucht das Blenden der Erscheinung, Die
    
    
    drängt! Verflucht, was uns in Träumen heuchelt, Des Ruhms, der Namensdauer Trug! Verflucht, was als Besitz uns schmeichelt, Als Weib und Kind, als Knecht und Pflug! Verflucht sei Mammon, wenn mit Schätzen Er uns zu kühnen Taten regt, Wenn er
    
    
    in Träumen heuchelt, Des Ruhms, der Namensdauer Trug! Verflucht, was als Besitz uns schmeichelt, Als Weib und Kind, als Knecht und Pflug! Verflucht sei Mammon, wenn mit Schätzen Er uns zu kühnen Taten regt, Wenn er zu müßigem Ergetzen Die
    
    
    Beginne, Mit hellem Sinne, Und neue Lieder Tönen darauf! Dies sind die Kleinen Von den Meinen. Höre, wie zu Lust und Taten Altklug sie raten! In die Welt weit, Aus der Einsamkeit, Wo Sinnen und Säfte stocken, Wollen sie dich
    
    
    den Meinen. Höre, wie zu Lust und Taten Altklug sie raten! In die Welt weit, Aus der Einsamkeit, Wo Sinnen und Säfte stocken, Wollen sie dich locken. Hör auf, mit deinem Gram zu spielen, Der, wie ein Geier, dir am
    
    
    Ein solcher Diener bringt Gefahr ins Haus. Ich will mich hier zu deinem Dienst verbinden, Auf deinen Wink nicht rasten und nicht ruhn; Wenn wir uns drüben wiederfinden, So sollst du mir das gleiche tun. Das Drüben kann mich wenig
    
    
    quillen meine Freuden, Und diese Sonne scheinet meinen Leiden; Kann ich mich erst von ihnen scheiden, Dann mag, was will und kann, geschehn. Davon will ich nichts weiter hören, Ob man auch künftig haßt und liebt, Und ob es auch
    
    
    ihnen scheiden, Dann mag, was will und kann, geschehn. Davon will ich nichts weiter hören, Ob man auch künftig haßt und liebt, Und ob es auch in jenen Sphären Ein Oben oder Unten gibt. In diesem Sinne kannst du's wagen.
    
    
    gern davon befreien? Beglückt, wer Treue rein im Busen trägt, Kein Opfer wird ihn je gereuen! Allein ein Pergament, beschrieben und beprägt, Ist ein Gespenst, vor dem sich alle scheuen. Das Wort erstirbt schon in der Feder, Was willst du
    
    
    jedes Wunder gleich bereit! Stürzen wir uns in das Rauschen der Zeit, Ins Rollen der Begebenheit! Da mag denn Schmerz und Genuß, Gelingen und Verdruß Mit einander wechseln, wie es kann; Nur rastlos betätigt sich der Mann. Euch ist kein
    
    
    bereit! Stürzen wir uns in das Rauschen der Zeit, Ins Rollen der Begebenheit! Da mag denn Schmerz und Genuß, Gelingen und Verdruß Mit einander wechseln, wie es kann; Nur rastlos betätigt sich der Mann. Euch ist kein Maß und Ziel
    
    
    Genuß, Gelingen und Verdruß Mit einander wechseln, wie es kann; Nur rastlos betätigt sich der Mann. Euch ist kein Maß und Ziel gesetzt. Beliebt's Euch, überall zu naschen, Im Fliehen etwas zu erhaschen, Bekomm' Euch wohl, was Euch ergetzt. Nur
    
    
    Beliebt's Euch, überall zu naschen, Im Fliehen etwas zu erhaschen, Bekomm' Euch wohl, was Euch ergetzt. Nur greift mir zu und seid nicht blöde! Du hörest ja, von Freud' ist nicht die Rede. Dem Taumel weih' ich mich, dem schmerzlichsten
    
    
    verschließen, Und was der ganzen Menschheit zugeteilt ist, Will ich in meinem innern Selbst genießen, Mit meinem Geist das Höchst' und Tiefste greifen, Ihr Wohl und Weh auf meinen Busen häufen, Und so mein eigen Selbst zu ihrem Selbst erweitern,
    
    
    Menschheit zugeteilt ist, Will ich in meinem innern Selbst genießen, Mit meinem Geist das Höchst' und Tiefste greifen, Ihr Wohl und Weh auf meinen Busen häufen, Und so mein eigen Selbst zu ihrem Selbst erweitern, Und, wie sie selbst, am
    
    
    gemacht! Er findet sich in einem ew'gen Glanze, Uns hat er in die Finsternis gebracht, Und euch taugt einzig Tag und Nacht. Allein ich will! Das läßt sich hören! Doch nur vor einem ist mir bang: Die Zeit ist kurz,
    
    
    Des Löwen Mut, Des Hirsches Schnelligkeit, Des Italieners feurig Blut, Des Nordens Dau'rbarkeit. Laßt ihn Euch das Geheimnis finden, Großmut und Arglist zu verbinden, Und Euch, mit warmen Jugendtrieben, Nach einem Plane zu verlieben. Möchte selbst solch einen Herren kennen,
    
    
    man die Sachen eben sieht; Wir müssen das gescheiter machen, Eh' uns des Lebens Freude flieht. Was Henker! freilich Händ' und Füße Und Kopf und H – –, die sind dein; Doch alles, was ich frisch genieße, Ist das drum
    
    
    sieht; Wir müssen das gescheiter machen, Eh' uns des Lebens Freude flieht. Was Henker! freilich Händ' und Füße Und Kopf und H – –, die sind dein; Doch alles, was ich frisch genieße, Ist das drum weniger mein? Wenn ich
    
    
    Ist das drum weniger mein? Wenn ich sechs Hengste zahlen kann, Sind ihre Kräfte nicht die meine? Ich renne zu und bin ein rechter Mann, Als hätt' ich vierundzwanzig Beine. Drum frisch! Laß alles Sinnen sein, Und grad' mit in
    
    
    das an? Wir gehen eben fort. Was ist das für ein Marterort? Was heißt das für ein Leben führen, Sich und die Jungens ennuyieren? Laß du das dem Herrn Nachbar Wanst! Was willst du dich das Stroh zu dreschen plagen?
    
    
    nicht möglich, ihn zu sehn. Der arme Knabe wartet lange, Der darf nicht ungetröstet gehn. Komm, gib mir deinen Rock und Mütze; Die Maske muß mir köstlich stehn. Nun überlaß es meinem Witze! Ich brauche nur ein Viertelstündchen Zeit; Indessen
    
    
    überlaß es meinem Witze! Ich brauche nur ein Viertelstündchen Zeit; Indessen mache dich zur schönen Fahrt bereit! Verachte nur Vernunft und Wissenschaft, Des Menschen allerhöchste Kraft, Laß nur in Blend- und Zauberwerken Dich von dem Lügengeist bestärken, So hab' ich
    
    
    Indessen mache dich zur schönen Fahrt bereit! Verachte nur Vernunft und Wissenschaft, Des Menschen allerhöchste Kraft, Laß nur in Blend- und Zauberwerken Dich von dem Lügengeist bestärken, So hab' ich dich schon unbedingt – Ihm hat das Schicksal einen Geist
    
    
    schlepp' ich durch das wilde Leben, Durch flache Unbedeutenheit, Er soll mir zappeln, starren, kleben, Und seiner Unersättlichkeit Soll Speis' und Trank vor gier'gen Lippen schweben; Er wird Erquickung sich umsonst erflehn, Und hätt' er sich auch nicht dem Teufel
    
    
    übergeben, Er müßte doch zugrunde gehn! Ich bin allhier erst kurze Zeit, Und komme voll Ergebenheit, Einen Mann zu sprechen und zu kennen, Den alle mir mit Ehrfurcht nennen. Eure Höflichkeit erfreut mich sehr! Ihr seht einen Mann wie andre
    
    
    Ihr Euch sonst schon umgetan? Ich bitt' Euch, nehmt Euch meiner an! Ich komme mit allem guten Mut, Leidlichem Geld und frischem Blut; Meine Mutter wollte mich kaum entfernen; Möchte gern was Rechts hieraußen lernen. Da seid Ihr eben recht
    
    
    gar beschränkter Raum, Man sieht nichts Grünes, keinen Baum, Und in den Sälen auf den Bänken Vergeht mir Hören, Sehn und Denken. Das kommt nur auf Gewohnheit an. So nimmt ein Kind der Mutter Brust Nicht gleich im Anfang willig
    
    
    wünschte recht gelehrt zu werden, Und möchte gern, was auf der Erden Und in dem Himmel ist, erfassen, Die Wissenschaft und die Natur. Da seid Ihr auf der rechten Spur; Doch müßt Ihr Euch nicht zerstreuen lassen. Ich bin dabei
    
    
    Natur. Da seid Ihr auf der rechten Spur; Doch müßt Ihr Euch nicht zerstreuen lassen. Ich bin dabei mit Seel' und Leib; Doch freilich würde mir behagen Ein wenig Freiheit und Zeitvertreib An schönen Sommerfeiertagen. Gebraucht der Zeit, sie geht
    
    
    Ihr Euch nicht zerstreuen lassen. Ich bin dabei mit Seel' und Leib; Doch freilich würde mir behagen Ein wenig Freiheit und Zeitvertreib An schönen Sommerfeiertagen. Gebraucht der Zeit, sie geht so schnell von hinnen, Doch Ordnung lehrt Euch Zeit gewinnen.
    
    
    Euch wohl dressiert, In spanische Stiefeln eingeschnürt, Daß er bedächtiger so fortan Hinschleiche die Gedankenbahn, Und nicht etwa, die Kreuz und Quer, Irrlichteliere hin und her. Dann lehret man Euch manchen Tag, Getrieben, wie Essen und Trinken frei, Eins! Zwei!
    
    
    spanische Stiefeln eingeschnürt, Daß er bedächtiger so fortan Hinschleiche die Gedankenbahn, Und nicht etwa, die Kreuz und Quer, Irrlichteliere hin und her. Dann lehret man Euch manchen Tag, Getrieben, wie Essen und Trinken frei, Eins! Zwei! Drei! dazu nötig sei.
    
    
    Und nicht etwa, die Kreuz und Quer, Irrlichteliere hin und her. Dann lehret man Euch manchen Tag, Getrieben, wie Essen und Trinken frei, Eins! Zwei! Drei! dazu nötig sei. Zwar ist's mit der Gedankenfabrik Wie mit einem Weber-Meisterstück, Wo ein
    
    
    tritt herein Und beweist Euch, es müßt' so sein: Das Erst' wär' so, das Zweite so, Und drum das Dritt' und Vierte so, Und wenn das Erst' und Zweit' nicht wär', Das Dritt' und Viert' wär' nimmermehr. Das preisen die
    
    
    so sein: Das Erst' wär' so, das Zweite so, Und drum das Dritt' und Vierte so, Und wenn das Erst' und Zweit' nicht wär', Das Dritt' und Viert' wär' nimmermehr. Das preisen die Schüler aller Orten, Sind aber keine Weber
    
    
    das Zweite so, Und drum das Dritt' und Vierte so, Und wenn das Erst' und Zweit' nicht wär', Das Dritt' und Viert' wär' nimmermehr. Das preisen die Schüler aller Orten, Sind aber keine Weber geworden. Wer will was Lebendigs erkennen
    
    
    und Viert' wär' nimmermehr. Das preisen die Schüler aller Orten, Sind aber keine Weber geworden. Wer will was Lebendigs erkennen und beschreiben, Sucht erst den Geist heraus zu treiben, Dann hat er die Teile in seiner Hand, Fehlt leider! nur
    
    
    er die Teile in seiner Hand, Fehlt leider! nur das geistige Band. Encheiresin naturae nennt's die Chemie, Spottet ihrer selbst und weiß nicht wie. Kann Euch nicht eben ganz verstehen. Das wird nächstens schon besser gehen, Wenn Ihr lernt alles
    
    
    die Metaphysik machen! Da seht, daß Ihr tiefsinnig faßt, Was in des Menschen Hirn nicht paßt; Für was drein geht und nicht drein geht, Ein prächtig Wort zu Diensten steht. Doch vorerst dieses halbe Jahr Nehmt ja der besten Ordnung
    
    
    kann es Euch so sehr nicht übel nehmen, Ich weiß, wie es um diese Lehre steht. Es erben sich Gesetz' und Rechte Wie eine ew'ge Krankheit fort, Sie schleppen von Geschlecht sich zum Geschlechte Und rücken sacht von Ort zu
    
    
    satt, Muß wieder recht den Teufel spielen. Laut. Der Geist der Medizin ist leicht zu fassen; Ihr durchstudiert die groß' und kleine Welt, Um es am Ende gehn zu lassen, Wie's Gott gefällt. Vergebens, daß Ihr ringsum wissenschaftlich schweift, Ein
    
    
    Ihr Euch nur selbst vertraut, Vertrauen Euch die andern Seelen. Besonders lernt die Weiber führen; Es ist ihr ewig Weh und Ach So tausendfach Aus einem Punkte zu kurieren, Und wenn Ihr halbweg ehrbar tut, Dann habt Ihr sie all'
    
    
    die schlanke Hüfte frei, Zu sehn, wie fest geschnürt sie sei. Das sieht schon besser aus! Man sieht doch, wo und wie. Grau, teurer Freund, ist alle Theorie, Und grün des Lebens goldner Baum. Ich schwör' Euch zu, mir ist's
    
    
    Gönn' Eure Gunst mir dieses Zeichen! Sehr wohl. Eritis sicut Deus scientes bonum et malum. Folg' nur dem alten Spruch und meiner Muhme, der Schlange, Dir wird gewiß einmal bei deiner Gottähnlichkeit bange! Wohin soll es nun gehn? Wohin es
    
    
    du dir vertraust, sobald weißt du zu leben. Wie kommen wir denn aus dem Haus? Wo hast du Pferde, Knecht und Wagen? Wir breiten nur den Mantel aus, Der soll uns durch die Lüfte tragen. Du nimmst bei diesem kühnen
    
    
    Ihr wollt es ja, man soll es sein! Zur Tür hinaus, wer sich entzweit! Mit offner Brust singt Runda, sauft und schreit! Auf! Holla! Ho! Weh mir, ich bin verloren! Baumwolle her! der Kerl sprengt mir die Ohren. Wenn das
    
    
    auf, Frau Nachtigall, Grüß' mir mein Liebchen zehentausendmal. Dem Liebchen keinen Gruß! ich will davon nichts hören! Dem Liebchen Gruß und Kuß! du wirst mir's nicht verwehren. Riegel auf! in stiller Nacht. Riegel auf! der Liebste wacht. Riegel zu! des
    
    
    verwehren. Riegel auf! in stiller Nacht. Riegel auf! der Liebste wacht. Riegel zu! des Morgens früh. Ja, singe, singe nur und lob' und rühme sie! Ich will zu meiner Zeit schon lachen. Sie hat mich angeführt, dir wird sie's auch
    
    
    auf! in stiller Nacht. Riegel auf! der Liebste wacht. Riegel zu! des Morgens früh. Ja, singe, singe nur und lob' und rühme sie! Ich will zu meiner Zeit schon lachen. Sie hat mich angeführt, dir wird sie's auch so machen.
    
    
    Bock, wenn er vom Blocksberg kehrt, Mag im Galopp noch gute Nacht ihr meckern! Ein braver Kerl von echtem Fleisch und Blut Ist für die Dirne viel zu gut. Ich will von keinem Gruße wissen, Als ihr die Fenster eingeschmissen!
    
    
    was zum besten geben. Und singt den Rundreim kräftig mit! Es war eine Ratt' im Kellernest, Lebte nur von Fett und Butter, Hatte sich ein Ränzlein angemäst't, Als wie der Doktor Luther. Die Köchin hatt' ihr Gift gestellt; Da ward's
    
    
    Als hätt' es Lieb' im Leibe. Sie kam für Angst am hellen Tag Der Küche zugelaufen, Fiel an den Herd und zuckt' und lag, Und tät erbärmlich schnaufen. Da lachte die Vergifterin noch: Ha! sie pfeift auf dem letzten Loch,
    
    
    es Lieb' im Leibe. Sie kam für Angst am hellen Tag Der Küche zugelaufen, Fiel an den Herd und zuckt' und lag, Und tät erbärmlich schnaufen. Da lachte die Vergifterin noch: Ha! sie pfeift auf dem letzten Loch, Als hätte
    
    
    zu streuen! Sie stehn wohl sehr in deiner Gunst? Der Schmerbauch mit der kahlen Platte! Das Unglück macht ihn zahm und mild; Er sieht in der geschwollnen Ratte Sein ganz natürlich Ebenbild. Ich muß dich nun vor allen Dingen In
    
    
    Katzen mit dem Schwanz. Wenn sie nicht über Kopfweh klagen, So lang' der Wirt nur weiter borgt, Sind sie vergnügt und unbesorgt. Die kommen eben von der Reise, Man sieht's an ihrer wunderlichen Weise; Sie sind nicht eine Stunde hier.
    
    
    Sie sind nicht eine Stunde hier. Wahrhaftig, du hast recht! Mein Leipzig lob' ich mir! Es ist ein klein Paris, und bildet seine Leute. Für was siehst du die Fremden an? Laßt mich nur gehn! Bei einem vollen Glase Zieh'
    
    
    einen Kinderzahn, Den Burschen leicht die Würmer aus der Nase. Sie scheinen mir aus einem edlen Haus, Sie sehen stolz und unzufrieden aus. Marktschreier sind's gewiß, ich wette! Vielleicht. Gib acht, ich schraube sie! Den Teufel spürt das Völkchen nie,
    
    
    ihr begehrt, die Menge. Nur auch ein nagelneues Stück! Wir kommen erst aus Spanien zurück, Dem schönen Land des Weins und der Gesänge. Es war einmal ein König, Der hatt' einen großen Floh – Horcht! Einen Floh! Habt ihr das
    
    
    er mir aufs genauste mißt, Und daß, so lieb sein Kopf ihm ist, Die Hosen keine Falten werfen! In Sammet und in Seide War er nun angetan, Hatte Bänder auf dem Kleide, Hatt' auch ein Kreuz daran, Und war sogleich
    
    
    Und hatt einen großen Stern. Da wurden seine Geschwister Bei Hof' auch große Herrn. Die waren sehr geplagt, Die Königin und die Zofe Gestochen und genagt, Und durften sie nicht knicken, Und weg sie jucken nicht. Wir knicken und ersticken
    
    
    Stern. Da wurden seine Geschwister Bei Hof' auch große Herrn. Die waren sehr geplagt, Die Königin und die Zofe Gestochen und genagt, Und durften sie nicht knicken, Und weg sie jucken nicht. Wir knicken und ersticken Doch gleich, wenn einer
    
    
    Die Königin und die Zofe Gestochen und genagt, Und durften sie nicht knicken, Und weg sie jucken nicht. Wir knicken und ersticken Doch gleich, wenn einer sticht. Wir knicken und ersticken Doch gleich, wenn einer sticht. Bravo! Bravo! Das war
    
    
    durften sie nicht knicken, Und weg sie jucken nicht. Wir knicken und ersticken Doch gleich, wenn einer sticht. Wir knicken und ersticken Doch gleich, wenn einer sticht. Bravo! Bravo! Das war schön! So soll es jedem Floh ergehn! Spitzt die
    
    
    ersticken Doch gleich, wenn einer sticht. Bravo! Bravo! Das war schön! So soll es jedem Floh ergehn! Spitzt die Finger und packt sie fein! Es lebe die Freiheit! Es lebe der Wein! Ich tränke gern ein Glas, die Freiheit hoch
    
    
    kann Wein auch geben. Ein tiefer Blick in die Natur! Hier ist ein Wunder, glaubet nur! Nun zieht die Pfropfen und genießt! O schöner Brunnen, der uns fließt! Nur hütet euch, daß ihr mir nichts vergießt! Uns ist ganz kannibalisch
    
    
    begegnen? Wart' nur, es sollen Schläge regnen! Ich brenne! ich brenne! Zauberei! Stoßt zu! der Kerl ist vogelfrei! Falsch Gebild und Wort Verändern Sinn und Ort! Seid hier und dort! Wo bin ich? Welches schöne Land! Weinberge! Seh' ich recht?
    
    
    sollen Schläge regnen! Ich brenne! ich brenne! Zauberei! Stoßt zu! der Kerl ist vogelfrei! Falsch Gebild und Wort Verändern Sinn und Ort! Seid hier und dort! Wo bin ich? Welches schöne Land! Weinberge! Seh' ich recht? Und Trauben gleich zur
    
    
    brenne! ich brenne! Zauberei! Stoßt zu! der Kerl ist vogelfrei! Falsch Gebild und Wort Verändern Sinn und Ort! Seid hier und dort! Wo bin ich? Welches schöne Land! Weinberge! Seh' ich recht? Und Trauben gleich zur Hand! Hier unter diesem
    
    
    sehn –– Es liegt mir bleischwer in den Füßen. Mein! Sollte wohl der Wein noch fließen? Betrug war alles, Lug und Schein. Mir deuchte doch, als tränk' ich Wein. Aber wie war es mit den Trauben? Nun sag' mir eins,
    
    
    Jahre mir vom Leibe? Weh mir, wenn du nichts Bessers weißt! Schon ist die Hoffnung mir verschwunden. Hat die Natur und hat ein edler Geist Nicht irgendeinen Balsam ausgefunden? Mein Freund, nun sprichst du wieder klug! Dich zu verjüngen, gibt's
    
    
    in einem andern Buch, Und ist ein wunderlich Kapitel. Ich will es wissen. Gut! Ein Mittel, ohne Geld Und Arzt und Zauberei zu haben: Begib dich gleich hinaus aufs Feld, Fang an zu hacken und zu graben, Erhalte dich und
    
    
    Ein Mittel, ohne Geld Und Arzt und Zauberei zu haben: Begib dich gleich hinaus aufs Feld, Fang an zu hacken und zu graben, Erhalte dich und deinen Sinn In einem ganz beschränkten Kreise, Ernähre dich mit ungemischter Speise, Leb mit
    
    
    Arzt und Zauberei zu haben: Begib dich gleich hinaus aufs Feld, Fang an zu hacken und zu graben, Erhalte dich und deinen Sinn In einem ganz beschränkten Kreise, Ernähre dich mit ungemischter Speise, Leb mit dem Vieh als Vieh, und
    
    
    dich und deinen Sinn In einem ganz beschränkten Kreise, Ernähre dich mit ungemischter Speise, Leb mit dem Vieh als Vieh, und acht es nicht für Raub, Den Acker, den du erntest, selbst zu düngen; Das ist das beste Mittel, glaub,
    
    
    du den Trank nicht selber brauen? Das wär' ein schöner Zeitvertreib! Ich wollt' indes wohl tausend Brücken bauen. Nicht Kunst und Wissenschaft allein, Geduld will bei dem Werke sein. Ein stiller Geist ist Jahre lang geschäftig, Die Zeit nur macht
    
    
    Wie glücklich würde sich der Affe schätzen, Könnt' er nur auch ins Lotto setzen! Das ist die Welt; Sie steigt und fällt Und rollt beständig; Sie klingt wie Glas – Wie bald bricht das! Ist hohl inwendig. Hier glänzt sie
    
    
    dem Throne, Den Zepter halt' ich hier, es fehlt nur noch die Krone. O sei doch so gut, Mit Schweiß und mit Blut Die Krone zu leimen! Nun ist es geschehn! Wir reden und sehn, Wir hören und reimen –
    
    
    O sei doch so gut, Mit Schweiß und mit Blut Die Krone zu leimen! Nun ist es geschehn! Wir reden und sehn, Wir hören und reimen – Weh mir! ich werde schier verrückt. Nun fängt mir an fast selbst der
    
    
    gut, Mit Schweiß und mit Blut Die Krone zu leimen! Nun ist es geschehn! Wir reden und sehn, Wir hören und reimen – Weh mir! ich werde schier verrückt. Nun fängt mir an fast selbst der Kopf zu schwanken. Und
    
    
    ist nur Spaß, Der Takt, du Aas, Zu deiner Melodei. Erkennst du mich? Gerippe! Scheusal du! Erkennst du deinen Herrn und Meister? Was hält mich ab, so schlag' ich zu, Zerschmettre dich und deine Katzengeister! Hast du vorm roten Wams
    
    
    mich? Gerippe! Scheusal du! Erkennst du deinen Herrn und Meister? Was hält mich ab, so schlag' ich zu, Zerschmettre dich und deine Katzengeister! Hast du vorm roten Wams nicht mehr Respekt? Kannst du die Hahnenfeder nicht erkennen? Hab' ich dies
    
    
    gesehen haben. Auch die Kultur, die alle Welt beleckt, Hat auf den Teufel sich erstreckt; Wo siehst du Hörner, Schweif und Klauen? Und was den Fuß betrifft, den ich nicht missen kann, Der würde mir bei Leuten schaden; Darum bedien'
    
    
    Der würde mir bei Leuten schaden; Darum bedien' ich mich, wie mancher junge Mann, Seit vielen Jahren falscher Waden. Sinn und Verstand verlier' ich schier, Seh' ich den Junker Satan wieder hier! Den Namen, Weib, verbitt' ich mir! Warum? Was
    
    
    Eins mach Zehn, Und Zwei laß gehn, Und Drei mach gleich, So bist du reich. Verlier die Vier! Aus Fünf und Sechs, So sagt die Hex', Mach Sieben und Acht, So ist's vollbracht: Und Neun ist Eins, Und Zehn ist
    
    
    Drei mach gleich, So bist du reich. Verlier die Vier! Aus Fünf und Sechs, So sagt die Hex', Mach Sieben und Acht, So ist's vollbracht: Und Neun ist Eins, Und Zehn ist keins. Das ist das Hexen-Einmaleins. Mich dünkt, die
    
    
    damit verloren, Denn ein vollkommner Widerspruch Bleibt gleich geheimnisvoll für Kluge wie für Toren. Mein Freund, die Kunst ist alt und neu. Es war die Art zu allen Zeiten, Durch Drei und Eins, und Eins und Drei So schwätzt und
    
    
    wie für Toren. Mein Freund, die Kunst ist alt und neu. Es war die Art zu allen Zeiten, Durch Drei und Eins, und Eins und Drei So schwätzt und lehrt man ungestört; Wer will sich mit den Narrn befassen? Gewöhnlich
    
    
    Toren. Mein Freund, die Kunst ist alt und neu. Es war die Art zu allen Zeiten, Durch Drei und Eins, und Eins und Drei So schwätzt und lehrt man ungestört; Wer will sich mit den Narrn befassen? Gewöhnlich glaubt der
    
    
    Freund, die Kunst ist alt und neu. Es war die Art zu allen Zeiten, Durch Drei und Eins, und Eins und Drei So schwätzt und lehrt man ungestört; Wer will sich mit den Narrn befassen? Gewöhnlich glaubt der Mensch, wenn
    
    
    alt und neu. Es war die Art zu allen Zeiten, Durch Drei und Eins, und Eins und Drei So schwätzt und lehrt man ungestört; Wer will sich mit den Narrn befassen? Gewöhnlich glaubt der Mensch, wenn er nur Worte hört,
    
    
    Mich dünkt, ich hör' ein ganzes Chor Von hunderttausend Narren sprechen. Genug, genug, o treffliche Sibylle! Gib deinen Trank herbei, und fülle Die Schale rasch bis an den Rand hinan; Denn meinem Freund wird dieser Trunk nicht schaden: Er ist
    
    
    guten Schluck getan. Nur frisch hinunter! Immer zu! Es wird dir gleich das Herz erfreuen. Bist mit dem Teufel du und du, Und willst dich vor der Flamme scheuen? Nun frisch hinaus! Du darfst nicht ruhn. Mög' Euch das Schlückchen
    
    
    auf Walpurgis sagen. Hier ist ein Lied! wenn Ihr's zuweilen singt, So werdet Ihr besondre Wirkung spüren. Komm nur geschwind und laß dich führen; Du mußt notwendig transpirieren, Den edlen Müßiggang lehr' ich hernach dich schätzen, Und bald empfindest du
    
    
    transpirieren, Den edlen Müßiggang lehr' ich hernach dich schätzen, Und bald empfindest du mit innigem Ergetzen, Wie sich Cupido regt und hin und wider springt. Laß mich nur schnell noch in den Spiegel schauen! Das Frauenbild war gar zu schön!
    
    
    edlen Müßiggang lehr' ich hernach dich schätzen, Und bald empfindest du mit innigem Ergetzen, Wie sich Cupido regt und hin und wider springt. Laß mich nur schnell noch in den Spiegel schauen! Das Frauenbild war gar zu schön! Nein! Nein!
    
    
    Du siehst, mit diesem Trank im Leibe, Bald Helenen in jedem Weibe. Mein schönes Fräulein, darf ich wagen, Meinen Arm und Geleit Ihr anzutragen? Bin weder Fräulein, weder schön, Kann ungeleitet nach Hause gehn. Beim Himmel, dieses Kind ist schön!
    
    
    ungeleitet nach Hause gehn. Beim Himmel, dieses Kind ist schön! So etwas hab' ich nie gesehn. Sie ist so sitt- und tugendreich, Und etwas schnippisch doch zugleich. Der Lippe Rot, der Wange Licht, Die Tage der Welt vergess' ich's nicht!
    
    
    immer an. Mein Herr Magister Lobesan, Lass' Er mich mit dem Gesetz in Frieden! Und das sag' ich Ihm kurz und gut: Wenn nicht das süße junge Blut Heut nacht in meinen Armen ruht, So sind wir um Mitternacht geschieden.
    
    
    nicht das süße junge Blut Heut nacht in meinen Armen ruht, So sind wir um Mitternacht geschieden. Bedenkt, was gehn und stehen mag! Ich brauche wenigstens vierzehn Tag', Nur die Gelegenheit auszuspüren. Hätt' ich nur sieben Stunden Ruh', Brauchte den
    
    
    genießen? Die Freud' ist lange nicht so groß, Als wenn Ihr erst herauf, herum, Durch allerlei Brimborium, Das Püppchen geknetet und zugericht't, Wie's lehret manche welsche Geschicht'. Hab' Appetit auch ohne das. Jetzt ohne Schimpf und ohne Spaß. Ich sag'
    
    
    allerlei Brimborium, Das Püppchen geknetet und zugericht't, Wie's lehret manche welsche Geschicht'. Hab' Appetit auch ohne das. Jetzt ohne Schimpf und ohne Spaß. Ich sag' Euch: mit dem schönen Kind Geht's ein- für allemal nicht geschwind. Mit Sturm ist da
    
    
    Schaff mir ein Halstuch von ihrer Brust, Ein Strumpfband meiner Liebeslust! Damit Ihr seht, daß ich Eurer Pein Will förderlich und dienstlich sein, Wollen wir keinen Augenblick verlieren, Will Euch noch heut in ihr Zimmer führen. Und soll sie sehn?
    
    
    dieser Armut welche Fülle! In diesem Kerker welche Seligkeit! O nimm mich auf, der du die Vorwelt schon Bei Freud' und Schmerz im offnen Arm empfangen! Wie oft, ach! hat an diesem Väterthron Schon eine Schar von Kindern rings gehangen!
    
    
    Liebchen hier, mit vollen Kinderwangen, Dem Ahnherrn fromm die welke Hand geküßt. Ich fühl', o Mädchen, deinen Geist Der Füll' und Ordnung um mich säuseln, Der mütterlich dich täglich unterweist, Den Teppich auf den Tisch dich reinlich breiten heißt, Sogar
    
    
    schwör' Euch, ihr vergehn die Sinnen; Ich tat Euch Sächelchen hinein, Um eine andre zu gewinnen. Zwar Kind ist Kind und Spiel ist Spiel. Ich weiß nicht, soll ich? Fragt Ihr viel? Meint Ihr vielleicht den Schatz zu wahren? Dann
    
    
    den Kopf, reib' an den Händen – Nur fort! geschwind! –, Um Euch das süße junge Kind Nach Herzens Wunsch und Will' zu wenden; Und Ihr seht drein, Als solltet Ihr in den Hörsaal hinein, Als stünden grau leibhaftig vor
    
    
    wenden; Und Ihr seht drein, Als solltet Ihr in den Hörsaal hinein, Als stünden grau leibhaftig vor Euch da Physik und Metaphysika! Nur fort! Es ist so schwül, so dumpfig hie, Und ist doch eben so warm nicht drauß. Es
    
    
    meine wären! Man sieht doch gleich ganz anders drein. Was hilft euch Schönheit, junges Blut? Das ist wohl alles schön und gut, Allein man läßt's auch alles sein; Man lobt euch halb mit Erbarmen. Nach Golde drängt, Am Golde hängt
    
    
    nie sich übergessen; Die Kirch' allein, meine lieben Frauen, Kann ungerechtes Gut verdauen. Das ist ein allgemeiner Brauch, Ein Jud' und König kann es auch. Strich drauf ein Spange, Kett' und Ring', Als wären's eben Pfifferling', Dankt' nicht weniger und
    
    
    ungerechtes Gut verdauen. Das ist ein allgemeiner Brauch, Ein Jud' und König kann es auch. Strich drauf ein Spange, Kett' und Ring', Als wären's eben Pfifferling', Dankt' nicht weniger und nicht mehr, Als ob's ein Korb voll Nüsse wär', Versprach
    
    
    Jud' und König kann es auch. Strich drauf ein Spange, Kett' und Ring', Als wären's eben Pfifferling', Dankt' nicht weniger und nicht mehr, Als ob's ein Korb voll Nüsse wär', Versprach ihnen allen himmlischen Lohn – Und sie waren sehr
    
    
    waren sehr erbaut davon. Und Gretchen? Sitzt nun unruhvoll, Weiß weder, was sie will noch soll, Denkt ans Geschmeide Tag und Nacht, Noch mehr an den, der's ihr gebracht. Des Liebchens Kummer tut mir leid. Schaff du ihr gleich ein
    
    
    gleich ein neu Geschmeid'! Am ersten war ja so nicht viel. O ja, dem Herrn ist alles Kinderspiel! Und mach, und richt's nach meinem Sinn! Häng dich an ihre Nachbarin! Sei, Teufel, doch nur nicht wie Brei, Und schaff einen
    
    
    Und schaff einen neuen Schmuck herbei! Ja, gnäd'ger Herr, von Herzen gerne. So ein verliebter Tor verpufft Euch Sonne, Mond und alle Sterne Zum Zeitvertreib dem Liebchen in die Luft. Gott verzeih's meinem lieben Mann, Er hat an mir nicht
    
    
    mir die Kniee nieder! Da find' ich so ein Kästchen wieder In meinem Schrein, von Ebenholz, Und Sachen herrlich ganz und gar, Weit reicher, als das erste war. Das muß Sie nicht der Mutter sagen; Tät's wieder gleich zur Beichte
    
    
    dem Spiegelglas vorüber, Wir haben unsre Freude dran; Und dann gibt's einen Anlaß, gibt's ein Fest, Wo man's so nach und nach den Leuten sehen läßt. Ein Kettchen erst, die Perle dann ins Ohr; Die Mutter sieht's wohl nicht, man
    
    
    dich für ein Fräulein hält. Ich bin ein armes junges Blut; Ach Gott! der Herr ist gar zu gut: Schmuck und Geschmeide sind nicht mein. Ach, es ist nicht der Schmuck allein; Sie hat ein Wesen, einen Blick so scharf!
    
    
    – Ich wollt', ich hätt' eine frohere Mär! Ich hoffe, Sie läßt mich's drum nicht büßen: Ihr Mann ist tot und läßt Sie grüßen. Ist tot? das treue Herz! O weh! Mein Mann ist tot! Ach, ich vergeh'! Ach! liebe
    
    
    An einer wohlgeweihten Stätte Zum ewig kühlen Ruhebette. Habt Ihr sonst nichts an mich zu bringen? Ja, eine Bitte, groß und schwer; Lass' Sie doch ja für ihn dreihundert Messen singen! Im übrigen sind meine Taschen leer. Was! nicht ein
    
    
    es tut mir herzlich leid; Allein er hat sein Geld wahrhaftig nicht verzettelt. Auch er bereute seine Fehler sehr, Ja, und bejammerte sein Unglück noch viel mehr. Ach! daß die Menschen so unglücklich sind! Gewiß, ich will für ihn manch
    
    
    letzten Zügen, Wenn ich nur halb ein Kenner bin. ›Ich hatte‹, sprach er, ›nicht zum Zeitvertreib zu gaffen, Erst Kinder, und dann Brot für sie zu schaffen, Und Brot im allerweitsten Sinn, Und konnte nicht einmal mein Teil in Frieden
    
    
    nicht einmal mein Teil in Frieden essen.‹ Hat er so aller Treu', so aller Lieb' vergessen, Der Plackerei bei Tag und Nacht! Nicht doch, er hat Euch herzlich dran gedacht. Er sprach: ›Als ich nun weg von Malta ging, Da
    
    
    er hat Euch herzlich dran gedacht. Er sprach: ›Als ich nun weg von Malta ging, Da betet' ich für Frau und Kinder brünstig; Uns war denn auch der Himmel günstig, Daß unser Schiff ein türkisch Fahrzeug fing, Das einen Schatz
    
    
    haben. Ein schönes Fräulein nahm sich seiner an, Als er in Napel fremd umherspazierte; Sie hat an ihm viel Lieb's und Treu's getan, Daß er's bis an sein selig Ende spürte. Der Schelm! der Dieb an seinen Kindern! Auch alles
    
    
    dieser Welt den andern! Es konnte kaum ein herziger Närrchen sein. Er liebte nur das allzuviele Wandern; Und fremde Weiber, und fremden Wein, Und das verfluchte Würfelspiel. Nun, nun, so konnt' es gehn und stehen, Wenn er Euch ungefähr so
    
    
    nur das allzuviele Wandern; Und fremde Weiber, und fremden Wein, Und das verfluchte Würfelspiel. Nun, nun, so konnt' es gehn und stehen, Wenn er Euch ungefähr so viel Von seiner Seite nachgesehen. Ich schwör' Euch zu, mit dem Beding Wechselt'
    
    
    Kind! Lebt wohl, ihr Fraun! Lebt wohl! O sagt mir doch geschwind! Ich möchte gern ein Zeugnis haben, Wo, wie und wann mein Schatz gestorben und begraben. Ich bin von je der Ordnung Freund gewesen, Möcht' ihn auch tot im
    
    
    Lebt wohl! O sagt mir doch geschwind! Ich möchte gern ein Zeugnis haben, Wo, wie und wann mein Schatz gestorben und begraben. Ich bin von je der Ordnung Freund gewesen, Möcht' ihn auch tot im Wochenblättchen lesen. Ja, gute Frau,
    
    
    ist Gretchen Euer. Heut' abend sollt Ihr sie bei Nachbar' Marthen sehn: Das ist ein Weib wie auserlesen Zum Kuppler- und Zigeunerwesen! So recht! Doch wird auch was von uns begehrt. Ein Dienst ist wohl des andern wert. Wir legen
    
    
    Ihr's nun! Ist es das erstemal in Eurem Leben, Daß Ihr falsch Zeugnis abgelegt? Habt Ihr von Gott, der Welt und was sich drin bewegt, Vom Menschen, was sich ihm in Kopf und Herzen regt, Definitionen nicht mit großer Kraft
    
    
    Zeugnis abgelegt? Habt Ihr von Gott, der Welt und was sich drin bewegt, Vom Menschen, was sich ihm in Kopf und Herzen regt, Definitionen nicht mit großer Kraft gegeben? Mit frecher Stirne, kühner Brust? Und wollt Ihr recht ins Innre
    
    
    Innre gehen, Habt Ihr davon, Ihr müßt es grad' gestehen, So viel als von Herrn Schwerdtleins Tod gewußt! Du bist und bleibst ein Lügner, ein Sophiste. Ja, wenn man's nicht ein bißchen tiefer wüßte. Denn morgen wirst, in allen Ehren,
    
    
    morgen wirst, in allen Ehren, Das arme Gretchen nicht betören Und alle Seelenlieb' ihr schwören? Und zwar von Herzen. Gut und schön! Dann wird von ewiger Treu' und Liebe, Von einzig überallmächt'gem Triebe – Wird das auch so von Herzen
    
    
    Gretchen nicht betören Und alle Seelenlieb' ihr schwören? Und zwar von Herzen. Gut und schön! Dann wird von ewiger Treu' und Liebe, Von einzig überallmächt'gem Triebe – Wird das auch so von Herzen gehn? Laß das! Es wird! – Wenn
    
    
    ewig, ewig nenne, Ist das ein teuflisch Lügenspiel? Ich hab' doch recht! Hör! merk dir dies – Ich bitte dich, und schone meine Lunge –: Wer recht behalten will und hat nur eine Zunge, Behält's gewiß. Und komm, ich hab'
    
    
    hab' doch recht! Hör! merk dir dies – Ich bitte dich, und schone meine Lunge –: Wer recht behalten will und hat nur eine Zunge, Behält's gewiß. Und komm, ich hab' des Schwätzens Überdruß, Denn du hast recht, vorzüglich weil
    
    
    schaffen müssen! Die Mutter ist gar zu genau. Und Ihr, mein Herr, Ihr reist so immer fort? Ach, daß Gewerb' und Pflicht uns dazu treiben! Mit wieviel Schmerz verläßt man manchen Ort, Und darf doch nun einmal nicht bleiben! In
    
    
    Schmerz verläßt man manchen Ort, Und darf doch nun einmal nicht bleiben! In raschen Jahren geht's wohl an, So um und um frei durch die Welt zu streifen; Doch kömmt die böse Zeit heran, Und sich als Hagestolz allein zum
    
    
    Freunde häufig, Sie sind verständiger, als ich bin. O Beste! glaube, was man so verständig nennt, Ist oft mehr Eitelkeit und Kurzsinn. Wie? Ach, daß die Einfalt, daß die Unschuld nie Sich selbst und ihren heil'gen Wert erkennt! Daß Demut,
    
    
    so verständig nennt, Ist oft mehr Eitelkeit und Kurzsinn. Wie? Ach, daß die Einfalt, daß die Unschuld nie Sich selbst und ihren heil'gen Wert erkennt! Daß Demut, Niedrigkeit, die höchsten Gaben Der liebevoll austeilenden Natur – Denkt Ihr an mich
    
    
    Wirtschaft ist nur klein, Und doch will sie versehen sein. Wir haben keine Magd; muß kochen, fegen, stricken Und nähn, und laufen früh und spat; Und meine Mutter ist in allen Stücken So akkurat! Nicht daß sie just so sehr
    
    
    klein, Und doch will sie versehen sein. Wir haben keine Magd; muß kochen, fegen, stricken Und nähn, und laufen früh und spat; Und meine Mutter ist in allen Stücken So akkurat! Nicht daß sie just so sehr sich einzuschränken hat;
    
    
    sehr sich einzuschränken hat; Wir könnten uns weit eh'r als andre regen: Mein Vater hinterließ ein hübsch Vermögen, Ein Häuschen und ein Gärtchen vor der Stadt. Doch hab' ich jetzt so ziemlich stille Tage; Mein Bruder ist Soldat, Ich hatte
    
    
    gern noch einmal alle Plage, So lieb war mir das Kind. Ein Engel, wenn dir's glich. Ich zog es auf, und herzlich liebt' es mich. Es war nach meines Vaters Tod geboren. Die Mutter gaben wir verloren, So elend wie
    
    
    Tod geboren. Die Mutter gaben wir verloren, So elend wie sie damals lag, Und sie erholte sich sehr langsam, nach und nach. Da konnte sie nun nicht dran denken, Das arme Würmchen selbst zu tränken, Und so erzog ich's ganz
    
    
    konnte sie nun nicht dran denken, Das arme Würmchen selbst zu tränken, Und so erzog ich's ganz allein, Mit Milch und Wasser; so ward's mein. Auf meinem Arm, in meinem Schoß War's freundlich, zappelte, ward groß. Du hast gewiß das
    
    
    ich's tränken, bald es zu mir legen, Bald, wenn's nicht schwieg, vom Bett aufstehn Und tänzelnd in der Kammer auf und nieder gehn, Und früh am Tage schon am Waschtrog stehn; Dann auf dem Markt und an dem Herde sorgen,
    
    
    tänzelnd in der Kammer auf und nieder gehn, Und früh am Tage schon am Waschtrog stehn; Dann auf dem Markt und an dem Herde sorgen, Und immer fort wie heut so morgen. Da geht's, mein Herr, nicht immer mutig zu;
    
    
    nichts gefunden? Hat sich das Herz nicht irgendwo gebunden? Das Sprichwort sagt: Ein eigner Herd, Ein braves Weib sind Gold und Perlen wert. Ich meine, ob Ihr niemals Lust bekommen? Man hat mich überall recht höflich aufgenommen. Ich wollte sagen:
    
    
    dich! Mich überläuft's! O schaudre nicht! Laß diesen Blick, Laß diesen Händedruck dir sagen, Was unaussprechlich ist: Sich hinzugeben ganz und eine Wonne Zu fühlen, die ewig sein muß! Ewig! – Ihr Ende würde Verzweiflung sein. Nein, kein Ende! Kein
    
    
    ewig sein muß! Ewig! – Ihr Ende würde Verzweiflung sein. Nein, kein Ende! Kein Ende! Die Nacht bricht an. Ja, und wir wollen fort. Ich bät' Euch, länger hier zu bleiben, Allein es ist ein gar zu böser Ort. Es
    
    
    zu böser Ort. Es ist, als hätte niemand nichts zu treiben Und nichts zu schaffen, Als auf des Nachbarn Schritt und Tritt zu gaffen, Und man kommt ins Gered', wie man sich immer stellt. Und unser Pärchen? Ist den Gang
    
    
    ihre tiefe Brust, Wie in den Busen eines Freunds, zu schauen. Du führst die Reihe der Lebendigen Vor mir vorbei, und lehrst mich meine Brüder Im stillen Busch, in Luft und Wasser kennen. Und wenn der Sturm im Walde braust
    
    
    schauen. Du führst die Reihe der Lebendigen Vor mir vorbei, und lehrst mich meine Brüder Im stillen Busch, in Luft und Wasser kennen. Und wenn der Sturm im Walde braust und knarrt, Die Riesenfichte stürzend Nachbaräste Und Nachbarstämme quetschend niederstreift,
    
    
    und lehrst mich meine Brüder Im stillen Busch, in Luft und Wasser kennen. Und wenn der Sturm im Walde braust und knarrt, Die Riesenfichte stürzend Nachbaräste Und Nachbarstämme quetschend niederstreift, Und ihrem Fall dumpf hohl der Hügel donnert, Dann führst
    
    
    Und ihrem Fall dumpf hohl der Hügel donnert, Dann führst du mich zur sichern Höhle, zeigst Mich dann mir selbst, und meiner eignen Brust Geheime tiefe Wunder öffnen sich. Besänftigend herüber, schweben mir Von Felsenwänden, aus dem feuchten Busch Der
    
    
    O daß dem Menschen nichts Vollkommnes wird, Empfind' ich nun. Du gabst zu dieser Wonne, Die mich den Göttern nah und näher bringt, Mir den Gefährten, den ich schon nicht mehr Entbehren kann, wenn er gleich, kalt und frech, Mich
    
    
    den Göttern nah und näher bringt, Mir den Gefährten, den ich schon nicht mehr Entbehren kann, wenn er gleich, kalt und frech, Mich vor mir selbst erniedrigt, und zu Nichts, Mit einem Worthauch, deine Gaben wandelt. Er facht in meiner
    
    
    den Gefährten, den ich schon nicht mehr Entbehren kann, wenn er gleich, kalt und frech, Mich vor mir selbst erniedrigt, und zu Nichts, Mit einem Worthauch, deine Gaben wandelt. Er facht in meiner Brust ein wildes Feuer Nach jenem schönen
    
    
    plagen. Nun, nun! ich lass' dich gerne ruhn, Du darfst mir's nicht im Ernste sagen. An dir Gesellen, unhold, barsch und toll, Ist wahrlich wenig zu verlieren. Den ganzen Tag hat man die Hände voll! Was ihm gefällt und was
    
    
    unhold, barsch und toll, Ist wahrlich wenig zu verlieren. Den ganzen Tag hat man die Hände voll! Was ihm gefällt und was man lassen soll, Kann man dem Herrn nie an der Nase spüren. Das ist so just der rechte
    
    
    nicht, so wärst du schon Von diesem Erdball abspaziert. Dich wie ein Schuhu zu versitzen? Was schlurfst aus dumpfem Moos und triefendem Gestein, Wie eine Kröte, Nahrung ein? Ein schöner, süßer Zeitvertreib! Dir steckt der Doktor noch im Leib. Verstehst
    
    
    würdest du es ahnen können, Du wärest Teufel gnug, mein Glück mir nicht zu gönnen. Ein überirdisches Vergnügen! In Nacht und Tau auf den Gebirgen liegen, Und Erd' und Himmel wonniglich umfassen, Zu einer Gottheit sich aufschwellen lassen, Der Erde
    
    
    gnug, mein Glück mir nicht zu gönnen. Ein überirdisches Vergnügen! In Nacht und Tau auf den Gebirgen liegen, Und Erd' und Himmel wonniglich umfassen, Zu einer Gottheit sich aufschwellen lassen, Der Erde Mark mit Ahnungsdrang durchwühlen, Alle sechs Tagewerk' im
    
    
    gesittet Pfui zu sagen. Man darf das nicht vor keuschen Ohren nennen, Was keusche Herzen nicht entbehren können. Und kurz und gut, ich gönn' Ihm das Vergnügen, Gelegentlich sich etwas vorzulügen; Doch lange hält Er das nicht aus. Du bist
    
    
    lange hält Er das nicht aus. Du bist schon wieder abgetrieben, Und, währt es länger, aufgerieben In Tollheit oder Angst und Graus! Genug damit! Dein Liebchen sitzt dadrinne, Und alles wird ihr eng und trüb. Du kommst ihr gar nicht
    
    
    es länger, aufgerieben In Tollheit oder Angst und Graus! Genug damit! Dein Liebchen sitzt dadrinne, Und alles wird ihr eng und trüb. Du kommst ihr gar nicht aus dem Sinne, Sie hat dich übermächtig lieb. Erst kam deine Liebeswut übergeflossen,
    
    
    süßen Leib Nicht wieder vor die halb verrückten Sinnen! Was soll es denn? Sie meint, du seist entflohn, Und halb und halb bist du es schon. Ich bin ihr nah, und wär' ich noch so fern, Ich kann sie nie
    
    
    soll es denn? Sie meint, du seist entflohn, Und halb und halb bist du es schon. Ich bin ihr nah, und wär' ich noch so fern, Ich kann sie nie vergessen, nie verlieren; Ja, ich beneide schon den Leib des
    
    
    Gar wohl, mein Freund! Ich hab' Euch oft beneidet Ums Zwillingspaar, das unter Rosen weidet. Entfliehe, Kuppler! Schön! Ihr schimpft, und ich muß lachen. Der Gott, der Bub und Mädchen schuf, Erkannte gleich den edelsten Beruf, Auch selbst Gelegenheit zu
    
    
    beneidet Ums Zwillingspaar, das unter Rosen weidet. Entfliehe, Kuppler! Schön! Ihr schimpft, und ich muß lachen. Der Gott, der Bub und Mädchen schuf, Erkannte gleich den edelsten Beruf, Auch selbst Gelegenheit zu machen. Nur fort, es ist ein großer Jammer!
    
    
    ihrer Brust erwarmen! Fühl' ich nicht immer ihre Not? Bin ich der Flüchtling nicht? der Unbehauste? Der Unmensch ohne Zweck und Ruh', Der wie ein Wassersturz von Fels zu Felsen brauste Begierig wütend nach dem Abgrund zu? Und seitwärts sie,
    
    
    geschehn! Mag ihr Geschick auf mich zusammenstürzen Und sie mit mir zugrunde gehn! Wie's wieder siedet, wieder glüht! Geh ein und tröste sie, du Tor! Wo so ein Köpfchen keinen Ausgang sieht, Stellt er sich gleich das Ende vor. Es
    
    
    hältst nicht viel davon. Laß das, mein Kind! Du fühlst, ich bin dir gut; Für meine Lieben ließ' ich Leib und Blut, Will niemand sein Gefühl und seine Kirche rauben. Das ist nicht recht, man muß dran glauben! Muß man?
    
    
    mein Kind! Du fühlst, ich bin dir gut; Für meine Lieben ließ' ich Leib und Blut, Will niemand sein Gefühl und seine Kirche rauben. Das ist nicht recht, man muß dran glauben! Muß man? Ach! wenn ich etwas auf dich
    
    
    steigen freundlich blickend Ewige Sterne nicht herauf? Schau' ich nicht Aug' in Auge dir, Und drängt nicht alles Nach Haupt und Herzen dir, Und webt in ewigem Geheimnis Unsichtbar sichtbar neben dir? Erfüll davon dein Herz, so groß es ist,
    
    
    dann, wie du willst, Nenn's Glück! Herz! Liebe! Gott! Ich habe keinen Namen Dafür! Gefühl ist alles; Name ist Schall und Rauch, Umnebelnd Himmelsglut. Das ist alles recht schön und gut; Ungefähr sagt das der Pfarrer auch, Nur mit ein
    
    
    Ich habe keinen Namen Dafür! Gefühl ist alles; Name ist Schall und Rauch, Umnebelnd Himmelsglut. Das ist alles recht schön und gut; Ungefähr sagt das der Pfarrer auch, Nur mit ein bißchen andern Worten. Es sagen's allerorten Alle Herzen unter
    
    
    Antipathie! Ich muß nun fort. Ach, kann ich nie Ein Stündchen ruhig dir am Busen hängen, Und Brust an Brust und Seel' in Seele drängen? Ach, wenn ich nur alleine schlief'! Ich ließ' dir gern heut nacht den Riegel offen;
    
    
    Herr Doktor wurden da katechisiert, Hoff', es soll Ihnen wohl bekommen. Die Mädels sind doch sehr interessiert, Ob einer fromm und schlicht nach altem Brauch. Sie denken: duckt er da, folgt er uns eben auch. Du Ungeheuer siehst nicht ein,
    
    
    Daß sie den liebsten Mann verloren halten soll. Du übersinnlicher sinnlicher Freier, Ein Mägdelein nasführet dich. Du Spottgeburt von Dreck und Feuer! Und die Physiognomie versteht sie meisterlich: In meiner Gegenwart wird's ihr, sie weiß nicht wie, Mein Mäskchen da
    
    
    Die hat sich endlich auch betört. Das ist das Vornehmtun! Wieso? Es stinkt! Sie füttert zwei, wenn sie nun ißt und trinkt. Ach! So ist's ihr endlich recht ergangen. Wie lange hat sie an dem Kerl gehangen! Das war ein
    
    
    So ist's ihr endlich recht ergangen. Wie lange hat sie an dem Kerl gehangen! Das war ein Spazieren, Auf Dorf und Tanzplatz Führen, Mußt' überall die Erste sein, Kurtesiert' ihr immer mit Pastetchen und Wein; Bild't' sich was auf ihre
    
    
    gehangen! Das war ein Spazieren, Auf Dorf und Tanzplatz Führen, Mußt' überall die Erste sein, Kurtesiert' ihr immer mit Pastetchen und Wein; Bild't' sich was auf ihre Schönheit ein, War doch so ehrlos, sich nicht zu schämen, Geschenke von ihm
    
    
    was auf ihre Schönheit ein, War doch so ehrlos, sich nicht zu schämen, Geschenke von ihm anzunehmen. War ein Gekos' und ein Geschleck'; Da ist denn auch das Blümchen weg! Das arme Ding! Bedauerst sie noch gar! Wenn unsereins am
    
    
    Wenn unsereins am Spinnen war, Uns nachts die Mutter nicht hinunterließ, Stand sie bei ihrem Buhlen süß, Auf der Türbank und im dunkeln Gang Ward ihnen keine Stunde zu lang. Da mag sie denn sich ducken nun, Im Sünderhemdchen Kirchbuß'
    
    
    ein armes Mägdlein fehlen! Wie konnt' ich über andrer Sünden Nicht Worte gnug der Zunge finden! Wie schien mir's schwarz, und schwärzt's noch gar, Mir's immer doch nicht schwarz gnug war, Und segnet' mich und tat so groß, Und bin
    
    
    Zunge finden! Wie schien mir's schwarz, und schwärzt's noch gar, Mir's immer doch nicht schwarz gnug war, Und segnet' mich und tat so groß, Und bin nun selbst der Sünde bloß! Doch – alles, was dazu mich trieb, Gott! war
    
    
    Mit tausend Schmerzen Blickst auf zu deines Sohnes Tod. Zum Vater blickst du, Und Seufzer schickst du Hinauf um sein' und deine Not. Wer fühlet, Wie wühlet Der Schmerz mir im Gebein? Was mein armes Herz hier banget, Was es
    
    
    Kammer Die Sonne früh herauf, Saß ich in allem Jammer In meinem Bett schon auf. Hilf! rette mich von Schmach und Tod! Ach neige, Du Schmerzenreiche, Dein Antlitz gnädig meiner Not! Wenn ich so saß bei einem Gelag, Wo mancher
    
    
    lebendig von der Stelle! Wie von dem Fenster dort der Sakristei Aufwärts der Schein des ew'gen Lämpchens flämmert Und schwach und schwächer seitwärts dämmert, Und Finsternis drängt ringsum bei! So sieht's in meinem Busen nächtig. Und mir ist's wie dem
    
    
    weiß mich trefflich mit der Polizei, Doch mit dem Blutbann schlecht mich abzufinden. Heraus! Heraus! Herbei ein Licht! Man schilt und rauft, man schreit und ficht. Da liegt schon einer tot! Die Mörder, sind sie denn entflohn? Wer liegt hier?
    
    
    der Polizei, Doch mit dem Blutbann schlecht mich abzufinden. Heraus! Heraus! Herbei ein Licht! Man schilt und rauft, man schreit und ficht. Da liegt schon einer tot! Die Mörder, sind sie denn entflohn? Wer liegt hier? Deiner Mutter Sohn. Allmächtiger!
    
    
    Mutter Sohn. Allmächtiger! welche Not! Ich sterbe! das ist bald gesagt Und bälder noch getan. Was steht ihr Weiber, heult und klagt? Kommt her und hört mich an! Mein Gretchen, sieh! du bist noch jung, Bist gar noch nicht gescheit
    
    
    Not! Ich sterbe! das ist bald gesagt Und bälder noch getan. Was steht ihr Weiber, heult und klagt? Kommt her und hört mich an! Mein Gretchen, sieh! du bist noch jung, Bist gar noch nicht gescheit genung, Machst deine Sachen
    
    
    die Schande wird geboren, Wird sie heimlich zur Welt gebracht, Und man zieht den Schleier der Nacht Ihr über Kopf und Ohren; Ja, man möchte sie gern ermorden. Wächst sie aber und macht sich groß, Dann geht sie auch bei
    
    
    man zieht den Schleier der Nacht Ihr über Kopf und Ohren; Ja, man möchte sie gern ermorden. Wächst sie aber und macht sich groß, Dann geht sie auch bei Tage bloß, Und ist doch nicht schöner geworden. Je häßlicher wird
    
    
    nicht mehr am Altar stehn! In einem schönen Spitzenkragen Dich nicht beim Tanze wohlbehagen! In eine finstre Jammerecken Unter Bettler und Krüppel dich verstecken Und, wenn dir dann auch Gott verzeiht, Auf Erden sein vermaledeit! Befehlt Eure Seele Gott zu
    
    
    dich sprachst der Ehre los, Gabst mir den schwersten Herzensstoß. Ich gehe durch den Todesschlaf Zu Gott ein als Soldat und brav. Wie anders, Gretchen, war dir's, Als du noch voll Unschuld Hier zum Altar tratst, Aus dem vergriffnen Büchelchen
    
    
    Pein hinüberschlief? Auf deiner Schwelle wessen Blut? – Und unter deinem Herzen Regt sich's nicht quillend schon Und ängstet dich und sich Mit ahnungsvoller Gegenwart? Weh! Weh! Wär' ich der Gedanken los, Die mir herüber und hinüber gehen Wider mich!
    
    
    quillend schon Und ängstet dich und sich Mit ahnungsvoller Gegenwart? Weh! Weh! Wär' ich der Gedanken los, Die mir herüber und hinüber gehen Wider mich! Dies irae, dies illa Solvet saeclum in favilla. Grimm faßt dich! Die Posaune tönt! Die
    
    
    Nil inultum remanebit. Mir wird so eng! Die Mauernpfeiler Befangen mich! Das Gewölbe Drängt mich! – Luft! Verbirg dich! Sünd' und Schande Bleibt nicht verborgen. Luft? Licht? Weh dir! Quid sum miser tunc dicturus? Quem patronum rogaturus? Cum vix justus
    
    
    nicht auch auf unsre Glieder wirken? Fürwahr, ich spüre nichts davon! Mir ist es winterlich im Leibe, Ich wünschte Schnee und Frost auf meiner Bahn. Wie traurig steigt die unvollkommne Scheibe Des roten Monds mit später Glut heran, Und leuchtet
    
    
    He da! mein Freund! darf ich dich zu uns fodern? Was willst du so vergebens lodern? Sei doch so gut und leucht' uns da hinauf! Aus Ehrfurcht, hoff' ich, soll es mir gelingen, Mein leichtes Naturell zu zwingen; Nur zickzack
    
    
    zaubertoll, Und wenn ein Irrlicht Euch die Wege weisen soll, So müßt Ihr's so genau nicht nehmen. In die Traum- und Zaubersphäre Sind wir, scheint es, eingegangen. Führ' uns gut und mach' dir Ehre, Daß wir vorwärts bald gelangen In
    
    
    So müßt Ihr's so genau nicht nehmen. In die Traum- und Zaubersphäre Sind wir, scheint es, eingegangen. Führ' uns gut und mach' dir Ehre, Daß wir vorwärts bald gelangen In den weiten, öden Räumen! Seh' die Bäume hinter Bäumen, Wie
    
    
    schnell vorüberrücken, Und die Klippen, die sich bücken, Und die langen Felsennasen, Wie sie schnarchen, wie sie blasen! Eilet Bach und Bächlein nieder. Hör' ich Rauschen? hör' ich Lieder? Hör' ich holde Liebesklage, Stimmen jener Himmelstage? Was wir hoffen, was
    
    
    hoffen, was wir lieben! Und das Echo, wie die Sage Alter Zeiten, hallet wider. Uhu! Schuhu! tönt es näher, Kauz und Kiebitz und der Häher, Sind sie alle wach geblieben? Sind das Molche durchs Gesträuche? Lange Beine, dicke Bäuche! Und
    
    
    wir lieben! Und das Echo, wie die Sage Alter Zeiten, hallet wider. Uhu! Schuhu! tönt es näher, Kauz und Kiebitz und der Häher, Sind sie alle wach geblieben? Sind das Molche durchs Gesträuche? Lange Beine, dicke Bäuche! Und die Wurzeln,
    
    
    geblieben? Sind das Molche durchs Gesträuche? Lange Beine, dicke Bäuche! Und die Wurzeln, wie die Schlangen, Winden sich aus Fels und Sande, Strecken wunderliche Bande, Uns zu schrecken, uns zu fangen; Aus belebten derben Masern Strecken sie Polypenfasern Nach dem
    
    
    zu fangen; Aus belebten derben Masern Strecken sie Polypenfasern Nach dem Wandrer. Und die Mäuse Tausendfärbig, scharenweise, Durch das Moos und durch die Heide! Und die Funkenwürmer fliegen Mit gedrängten Schwärmezügen Zum verwirrenden Geleite. Aber sag' mir, ob wir stehen,
    
    
    Schwärmezügen Zum verwirrenden Geleite. Aber sag' mir, ob wir stehen, Oder ob wir weitergehen? Alles, alles scheint zu drehen, Fels und Bäume, die Gesichter Schneiden, und die irren Lichter, Die sich mehren, die sich blähen. Fasse wacker meinen Zipfel! Hier
    
    
    sag' mir, ob wir stehen, Oder ob wir weitergehen? Alles, alles scheint zu drehen, Fels und Bäume, die Gesichter Schneiden, und die irren Lichter, Die sich mehren, die sich blähen. Fasse wacker meinen Zipfel! Hier ist so ein Mittelgipfel, Wo
    
    
    Schein! Und selbst bis in die tiefen Schlünde Da steigt ein Dampf, dort ziehen Schwaden, Hier leuchtet Glut aus Dunst und Flor, Dann schleicht sie wie ein zarter Faden, Dann bricht sie wie ein Quell hervor. Hier schlingt sie eine
    
    
    Nacht. Höre, wie's durch die Wälder kracht! Aufgescheucht fliegen die Eulen. Hör', es splittern die Säulen Ewig grüner Paläste. Girren und Brechen der Äste! Der Stämme mächtiges Dröhnen! Der Wurzeln Knarren und Gähnen! Im fürchterlich verworrenen Falle Übereinander krachen sie
    
    
    Hör', es splittern die Säulen Ewig grüner Paläste. Girren und Brechen der Äste! Der Stämme mächtiges Dröhnen! Der Wurzeln Knarren und Gähnen! Im fürchterlich verworrenen Falle Übereinander krachen sie alle, Und durch die übertrümmerten Klüfte Zischen und heulen die Lüfte.
    
    
    Dröhnen! Der Wurzeln Knarren und Gähnen! Im fürchterlich verworrenen Falle Übereinander krachen sie alle, Und durch die übertrümmerten Klüfte Zischen und heulen die Lüfte. Hörst du Stimmen in der Höhe? In der Ferne, in der Nähe? Ja, den ganzen Berg
    
    
    Brocken ziehn, Die Stoppel ist gelb, die Saat ist grün. Herr Urian sitzt oben auf. So geht es über Stein und Stock, Es f–t die Hexe, es stinkt der Bock. Die alte Baubo kommt allein, Sie reitet auf einem Mutterschwein.
    
    
    Bock. Die alte Baubo kommt allein, Sie reitet auf einem Mutterschwein. So Ehre denn, wem Ehre gebührt! Frau Baubo vor! und angeführt! Ein tüchtig Schwein und Mutter drauf, Da folgt der ganze Hexenhauf. Welchen Weg kommst du her? Übern Ilsenstein!
    
    
    allein, Sie reitet auf einem Mutterschwein. So Ehre denn, wem Ehre gebührt! Frau Baubo vor! und angeführt! Ein tüchtig Schwein und Mutter drauf, Da folgt der ganze Hexenhauf. Welchen Weg kommst du her? Übern Ilsenstein! Da guckt' ich der Eule
    
    
    einem Sprunge macht's der Mann. Kommt mit, kommt mit, vom Felsensee! Wir möchten gerne mit in die Höh'. Wir waschen, und blank sind wir ganz und gar; Aber auch ewig unfruchtbar. Es schweigt der Wind, es flieht der Stern, Der
    
    
    Kommt mit, kommt mit, vom Felsensee! Wir möchten gerne mit in die Höh'. Wir waschen, und blank sind wir ganz und gar; Aber auch ewig unfruchtbar. Es schweigt der Wind, es flieht der Stern, Der trübe Mond verbirgt sich gern.
    
    
    nicht flog. Und wenn wir um den Gipfel ziehn, So streichet an dem Boden hin, Und deckt die Heide weit und breit Mit eurem Schwarm der Hexenheit. Das drängt und stößt, das ruscht und klappert! Das zischt und quirlt, das
    
    
    So streichet an dem Boden hin, Und deckt die Heide weit und breit Mit eurem Schwarm der Hexenheit. Das drängt und stößt, das ruscht und klappert! Das zischt und quirlt, das zieht und plappert! Das leuchtet, sprüht und stinkt und
    
    
    Boden hin, Und deckt die Heide weit und breit Mit eurem Schwarm der Hexenheit. Das drängt und stößt, das ruscht und klappert! Das zischt und quirlt, das zieht und plappert! Das leuchtet, sprüht und stinkt und brennt! Ein wahres Hexenelement!
    
    
    die Heide weit und breit Mit eurem Schwarm der Hexenheit. Das drängt und stößt, das ruscht und klappert! Das zischt und quirlt, das zieht und plappert! Das leuchtet, sprüht und stinkt und brennt! Ein wahres Hexenelement! Nur fest an mir!
    
    
    breit Mit eurem Schwarm der Hexenheit. Das drängt und stößt, das ruscht und klappert! Das zischt und quirlt, das zieht und plappert! Das leuchtet, sprüht und stinkt und brennt! Ein wahres Hexenelement! Nur fest an mir! sonst sind wir gleich
    
    
    Hexenheit. Das drängt und stößt, das ruscht und klappert! Das zischt und quirlt, das zieht und plappert! Das leuchtet, sprüht und stinkt und brennt! Ein wahres Hexenelement! Nur fest an mir! sonst sind wir gleich getrennt. Wo bist du? Hier!
    
    
    drängt und stößt, das ruscht und klappert! Das zischt und quirlt, das zieht und plappert! Das leuchtet, sprüht und stinkt und brennt! Ein wahres Hexenelement! Nur fest an mir! sonst sind wir gleich getrennt. Wo bist du? Hier! Was! dort
    
    
    schon hingerissen? Da werd' ich Hausrecht brauchen müssen. Platz! Junker Voland kommt. Platz! süßer Pöbel, Platz! Hier, Doktor, fasse mich! und nun, in einem Satz, Laß uns aus dem Gedräng' entweichen; Es ist zu toll, sogar für meinesgleichen. Dort neben
    
    
    ein muntrer Klub beisammen. Im Kleinen ist man nicht allein. Doch droben möcht' ich lieber sein! Schon seh' ich Glut und Wirbelrauch. Dort strömt die Menge zu dem Bösen; Da muß sich manches Rätsel lösen. Doch manches Rätsel knüpft sich
    
    
    Es ist doch lange hergebracht, Daß in der großen Welt man kleine Welten macht. Da seh' ich junge Hexchen nackt und bloß, Und alte, die sich klug verhüllen. Seid freundlich, nur um meinetwillen; Die Müh' ist klein, der Spaß ist
    
    
    tönen! Verflucht Geschnarr! Man muß sich dran gewöhnen. Komm mit! Komm mit! Es kann nicht anders sein, Ich tret' heran und führe dich herein, Und ich verbinde dich aufs neue. Was sagst du, Freund? das ist kein kleiner Raum. Da
    
    
    ich auch will, verleugn' ich hier mich nicht. Komm nur! von Feuer gehen wir zu Feuer, Ich bin der Werber, und du bist der Freier. Ihr alten Herrn, was macht ihr hier am Ende? Ich lobt' euch, wenn ich euch
    
    
    was macht ihr hier am Ende? Ich lobt' euch, wenn ich euch hübsch in der Mitte fände, Von Saus umzirkt und Jugendbraus; Genug allein ist jeder ja zu Haus. Wer mag auf Nationen trauen, Man habe noch so viel für
    
    
    Zeit. Wir waren wahrlich auch nicht dumm, Und taten oft, was wir nicht sollten; Doch jetzo kehrt sich alles um und um, Und eben da wir's fest erhalten wollten. Wer mag wohl überhaupt jetzt eine Schrift Von mäßig klugem Inhalt
    
    
    doch ist nichts in meinem Laden, Dem keiner auf der Erde gleicht, Das nicht einmal zum tücht'gen Schaden Der Menschen und der Welt gereicht. Kein Dolch ist hier, von dem nicht Blut geflossen, Verzehrend heißes Gift ergossen, Kein Schmuck, der
    
    
    nicht selbst vergesse! Heiß' ich mir das doch eine Messe! Der ganze Strudel strebt nach oben; Du glaubst zu schieben und du wirst geschoben. Wer ist denn das? Betrachte sie genau! Lilith ist das. Wer? Adams erste Frau. Nimm dich
    
    
    das ist unerhört. Verschwindet doch! Wir haben ja aufgeklärt! Das Teufelspack, es fragt nach keiner Regel. Wir sind so klug, und dennoch spukt's in Tegel. Wie lange hab' ich nicht am Wahn hinausgekehrt, Und nie wird's rein; das ist doch
    
    
    will mir nichts gelingen; Doch eine Reise nehm' ich immer mit Und hoffe noch, vor meinem letzten Schritt, Die Teufel und die Dichter zu bezwingen. Er wird sich gleich in eine Pfütze setzen, Das ist die Art, wie er sich
    
    
    Das ist die Art, wie er sich soulagiert, Und wenn Blutegel sich an seinem Steiß ergetzen, Ist er von Geistern und von Geist kuriert. Was lässest du das schöne Mädchen fahren, Das dir zum Tanz so lieblich sang? Ach! mitten
    
    
    Wer fragt darnach in einer Schäferstunde? Dann sah ich – Was? Mephisto, siehst du dort Ein blasses, schönes Kind allein und ferne stehen? Sie schiebt sich langsam nur vom Ort, Sie scheint mit geschloßnen Füßen zu gehen. Ich muß bekennen,
    
    
    Blocksberg finde, Das find' ich gut; denn da gehört ihr hin. Heute ruhen wir einmal, Miedings wackre Söhne. Alter Berg und feuchtes Tal, Das ist die ganze Szene! Daß die Hochzeit golden sei, Solln funfzig Jahr sein vorüber; Aber ist
    
    
    der Streit vorbei, Das Golden ist mir lieber. Seid ihr Geister, wo ich bin, So zeigt's in diesen Stunden; König und die Königin, Sie sind aufs neu verbunden. Kommt der Puck und dreht sich quer Und schleift den Fuß im
    
    
    wo ich bin, So zeigt's in diesen Stunden; König und die Königin, Sie sind aufs neu verbunden. Kommt der Puck und dreht sich quer Und schleift den Fuß im Reihen, Sich auch mit ihm zu freuen. Ariel bewegt den Sang
    
    
    vertragen wollen, Lernen's von uns beiden! Wenn sich zweie lieben sollen, Braucht man sie nur zu scheiden. Schmollt der Mann und grillt die Frau, So faßt sie nur behende, Führt mir nach dem Mittag Sie, Und Ihn an Nordens Ende.
    
    
    grillt die Frau, So faßt sie nur behende, Führt mir nach dem Mittag Sie, Und Ihn an Nordens Ende. Fliegenschnauz' und Mückennas' Mit ihren Anverwandten, Frosch im Laub und Grill' im Gras, Das sind die Musikanten! Seht, da kommt der
    
    
    Führt mir nach dem Mittag Sie, Und Ihn an Nordens Ende. Fliegenschnauz' und Mückennas' Mit ihren Anverwandten, Frosch im Laub und Grill' im Gras, Das sind die Musikanten! Seht, da kommt der Dudelsack! Es ist die Seifenblase. Hört den Schneckeschnickeschnack
    
    
    kommt der Dudelsack! Es ist die Seifenblase. Hört den Schneckeschnickeschnack Durch seine stumpfe Nase. GEIST, DER SICH ERST BILDET. Spinnenfuß und Krötenbauch Und Flügelchen dem Wichtchen! Zwar ein Tierchen gibt es nicht, Doch gibt es ein Gedichtchen. Kleiner Schritt und
    
    
    Spinnenfuß und Krötenbauch Und Flügelchen dem Wichtchen! Zwar ein Tierchen gibt es nicht, Doch gibt es ein Gedichtchen. Kleiner Schritt und hoher Sprung Durch Honigtau und Düfte; Zwar du trippelst mir genung, Doch geht's nicht in die Lüfte. Ist das
    
    
    dem Wichtchen! Zwar ein Tierchen gibt es nicht, Doch gibt es ein Gedichtchen. Kleiner Schritt und hoher Sprung Durch Honigtau und Düfte; Zwar du trippelst mir genung, Doch geht's nicht in die Lüfte. Ist das nicht Maskeraden-Spott? Soll ich den
    
    
    hier geludert! Und von dem ganzen Hexenheer Sind zweie nur gepudert. Der Puder ist so wie der Rock Für alt' und graue Weibchen; Drum sitz' ich nackt auf meinem Bock Und zeig' ein derbes Leibchen. Wir haben zu viel Lebensart,
    
    
    ein derbes Leibchen. Wir haben zu viel Lebensart, Um hier mit euch zu maulen, Doch, hoff' ich, sollt ihr jung und zart, So wie ihr seid, verfaulen. Fliegenschnauz und Mückennas', Umschwärmt mir nicht die Nackte! Frosch im Laub und Grill'
    
    
    Um hier mit euch zu maulen, Doch, hoff' ich, sollt ihr jung und zart, So wie ihr seid, verfaulen. Fliegenschnauz und Mückennas', Umschwärmt mir nicht die Nackte! Frosch im Laub und Grill' im Gras, So bleibt doch auch im Takte!
    
    
    ihr jung und zart, So wie ihr seid, verfaulen. Fliegenschnauz und Mückennas', Umschwärmt mir nicht die Nackte! Frosch im Laub und Grill' im Gras, So bleibt doch auch im Takte! Gesellschaft wie man wünschen kann. Wahrhaftig lauter Bräute! Und Junggesellen,
    
    
    glaub'n sich nah dem Schatze. Auf Teufel reimt der Zweifel nur, Da bin ich recht am Platze. Frosch im Laub und Grill' im Gras, Verfluchte Dilettanten! Fliegenschnauz' und Mückennas', Ihr seid doch Musikanten! Sanssouci, so heißt das Heer Von lustigen
    
    
    reimt der Zweifel nur, Da bin ich recht am Platze. Frosch im Laub und Grill' im Gras, Verfluchte Dilettanten! Fliegenschnauz' und Mückennas', Ihr seid doch Musikanten! Sanssouci, so heißt das Heer Von lustigen Geschöpfen; Auf den Füßen geht's nicht mehr,
    
    
    erst entstanden; Doch sind wir gleich im Reihen hier Die glänzenden Galanten. Aus der Höhe schoß ich her Im Stern- und Feuerscheine, Liege nun im Grase quer – Wer hilft mir auf die Beine? Platz und Platz! und ringsherum! So
    
    
    schoß ich her Im Stern- und Feuerscheine, Liege nun im Grase quer – Wer hilft mir auf die Beine? Platz und Platz! und ringsherum! So gehn die Gräschen nieder, Geister kommen, Geister auch Sie haben plumpe Glieder. Tretet nicht so
    
    
    her Im Stern- und Feuerscheine, Liege nun im Grase quer – Wer hilft mir auf die Beine? Platz und Platz! und ringsherum! So gehn die Gräschen nieder, Geister kommen, Geister auch Sie haben plumpe Glieder. Tretet nicht so mastig auf
    
    
    Derbe, selber. Gab die liebende Natur, Gab der Geist euch Flügel, Folget meiner leichten Spur, Auf zum Rosenhügel! Pianissimo. Wolkenzug und Nebelflor Erhellen sich von oben. Luft im Laub und Wind im Rohr, Und alles ist zerstoben. Sie ist die
    
    
    euch Flügel, Folget meiner leichten Spur, Auf zum Rosenhügel! Pianissimo. Wolkenzug und Nebelflor Erhellen sich von oben. Luft im Laub und Wind im Rohr, Und alles ist zerstoben. Sie ist die Erste nicht. Endigst du? Rette sie! oder weh dir!
    
    
    Rette sie! oder weh dir! Den gräßlichsten Fluch über dich auf Jahrtausende! Bringe mich hin! Sie soll frei sein! Auf und davon! Was weben die dort um den Rabenstein? Weiß nicht, was sie kochen und schaffen. Schweben auf, schweben ab,
    
    
    hin! Sie soll frei sein! Auf und davon! Was weben die dort um den Rabenstein? Weiß nicht, was sie kochen und schaffen. Schweben auf, schweben ab, neigen sich, beugen sich. Eine Hexenzunft. Sie streuen und weihen. Vorbei! Vorbei! Mich faßt
    
    
    Rabenstein? Weiß nicht, was sie kochen und schaffen. Schweben auf, schweben ab, neigen sich, beugen sich. Eine Hexenzunft. Sie streuen und weihen. Vorbei! Vorbei! Mich faßt ein längst entwohnter Schauer, Der Menschheit ganzer Jammer faßt mich an. Hier wohnt sie,
    
    
    dem Schlafe schreien! Wer hat dir, Henker, diese Macht Über mich gegeben! Du holst mich schon um Mitternacht. Erbarme dich und laß mich leben! Ist's morgen früh nicht zeitig genung? Bin ich doch noch so jung, so jung! Und soll
    
    
    früh nicht zeitig genung? Bin ich doch noch so jung, so jung! Und soll schon sterben! Schön war ich auch, und das war mein Verderben. Nah war der Freund, nun ist er weit; Zerrissen liegt der Kranz, die Blumen zerstreut.
    
    
    seinen Hals will ich fliegen, An seinem Busen liegen! Er rief: Gretchen! Er stand auf der Schwelle. Mitten durchs Heulen und Klappen der Hölle, Durch den grimmigen, teuflischen Hohn Erkannt' ich den süßen, den liebenden Ton. Ich bin's! Du bist's!
    
    
    Schon ist die Straße wieder da, Auf der ich dich zum ersten Male sah. Und der heitere Garten, Wo ich und Marthe deiner warten. Komm mit! Komm mit! O weile! Weil' ich doch so gern, wo du weilest Eile! Wenn
    
    
    komm! schon weicht die tiefe Nacht. Meine Mutter hab' ich umgebracht, Mein Kind hab' ich ertränkt. War es nicht dir und mir geschenkt? Dir auch. – Du bist's! ich glaub' es kaum. Gib deine Hand! Es ist kein Traum! Deine
    
    
    Mir ist's, als müßt' ich mich zu dir zwingen, Als stießest du mich von dir zurück; Und doch bist du's und blickst so gut, so komm! Fühlst du, daß ich es bin, so komm! Dahinaus? Ins Freie. Ist das Grab
    
    
    sie nicht. Der Platz, die Gassen Können sie nicht fassen. Die Glocke ruft, das Stäbchen bricht. Wie sie mich binden und packen! Zum Blutstuhl bin ich schon entrückt. Schon zuckt nach jedem Nacken Die Schärfe, die nach meinem zückt. Stumm
    
    
    Stumm liegt die Welt wie das Grab! O wär' ich nie geboren! Auf! oder ihr seid verloren. Unnützes Zagen! Zaudern und Plaudern! Meine Pferde schaudern, Der Morgen dämmert auf. Was steigt aus dem Boden herauf? Der! der! Schick' ihn fort!
    
    



```python
from tensorflow.keras.models import load_model
```


```python
model = load_model('trainedLSTMModel.h5')
```


```python
tokenizer = load(open('trainedLSTMModel', 'rb'))
```


```python
generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=50)
```




    "Trieb hab ' ich mich der Studien beflissen Zwar weiß ich ist ein gar zu böser Sterne auf ! O müßt Ihr ein moralisch Lied ! Er rief ein schönstes Müßiggang fehlen ! Es klopft . Ach Gott ! mag das ? Wer soll nicht 's . Sie küßt darnach"


