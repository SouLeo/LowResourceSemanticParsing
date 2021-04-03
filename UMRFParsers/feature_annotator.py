# from nltk.parse.corenlp import CoreNLPServer
from pycorenlp import StanfordCoreNLP
import re
import os
import ExtractTrainingDescriptions

class FeatureAnnotator:

    NumberTexts = ('hundred', 'thousand', 'million', 'billion', 'trillion',
                   'hundreds', 'thousands', 'millions', 'billions', 'trillions')
    DashedNumbers = re.compile(r'-*\d+-\d+')

    def __init__(self):
        working_directory = os.getcwd()
        core_nlp_directory = os.path.join(working_directory, 'stanford-corenlp-4.2.0')

        self.examples = ExtractTrainingDescriptions.extract_training_exs()

        self.nlp = StanfordCoreNLP('http://localhost:9000')
        self.nlp_properties = {
            'annotators': "tokenize,ssplit,pos,lemma,ner",
            "tokenize.options": "splitHyphenated=true,normalizeParentheses=false",
            "tokenize.whitespace": False,
            'ssplit.isOneSentence': True,
            'outputFormat': 'json'
        }

        self.annotated_examples = []
        self.tokens = []
        self.lemmas = []
        self.pos_tags = []
        self.ner_tags = []
        for example in self.examples:
            self.annotated_examples.append(self.annotate(example))
            self.tokens.append(self.annotate(example)['tokens'])
            self.lemmas.append(self.annotate(example)['lemmas'])
            self.pos_tags.append(self.annotate(example)['pos_tags'])
            self.ner_tags.append(self.annotate(example)['ner_tags'])




        self.server_properties = {
            'annotators': "tokenize,ssplit,pos,lemma,ner",
            "tokenize.options": "splitHyphenated=true,normalizeParentheses=false",
            "tokenize.whitespace": False,
            'ssplit.isOneSentence': True,
            'outputFormat': 'json'
        }

    @staticmethod
    def assert_equal_length(annotation):
        tokens = annotation['tokens']
        for key in annotation:
            if key == 'tokens':
                continue
            value = annotation[key]
            assert len(tokens) == len(value), (
                len(tokens), len(value), '\n', list(zip(tokens, value)), tokens, value)

    def annotate(self, text):
        tokens = self.nlp.annotate(text.strip(), self.nlp_properties)['sentences'][0]['tokens']
        output = dict(
            tokens=[], lemmas=[], pos_tags=[], ner_tags=[]
        )
        for token in tokens:
            output['tokens'].append(token['word'])
            output['lemmas'].append(token['lemma'])
            output['pos_tags'].append(token['pos'])
            output['ner_tags'].append(token['ner'])
        return output

    def __call__(self, text):
        annotation = self.annotate(text)
        annotation = self._combine_compounds(annotation)
        annotation = self._combine_numbers(annotation)
        annotation = self._tag_url_and_split_number(annotation)
        return annotation

    def _combine_compounds(self, annotation):
        # Combine tokens in compounds, e.g., 'make up' -> 'make-up'.
        tokens = []
        lemmas = []
        pos_tags = []
        ner_tags = []
        skip = False
        for i, lemma in enumerate(annotation['lemmas']):
            if skip:
                skip = False
            elif len(lemmas) > 0 and lemma in self.compound_map.get(lemmas[-1], []):
                # lemma belongs to a compound.
                lemmas[-1] = lemmas[-1] + '-' + lemma
                tokens[-1] = tokens[-1] + "-" + annotation['tokens'][i]
                pos_tags[-1] = "COMP"
                ner_tags[-1] = "0"
            elif len(lemmas) > 0 and lemma == "-" and i < len(annotation['lemmas']) - 1 \
                and annotation['lemmas'][i + 1] in self.compound_map.get(lemmas[-1], []):
                # lemma is a dash and the next lemma belongs to a compound.
                lemmas[-1] = lemmas[-1] + '-' + annotation['lemmas'][i + 1]
                tokens[-1] = tokens[-1] + '-' + annotation['tokens'][i + 1]
                pos_tags[-1] = "COMP"
                ner_tags[-1] = "0"
                skip = True # skip the next lemma.
            else:
                lemmas.append(lemma)
                tokens.append(annotation['tokens'][i])
                pos_tags.append(annotation['pos_tags'][i])
                ner_tags.append(annotation['ner_tags'][i])

        output = dict(
            tokens=tokens, lemmas=lemmas, pos_tags=pos_tags, ner_tags=ner_tags
        )
        self.assert_equal_length(output)
        return output

    def _combine_numbers(self, annotation):

        def two_combinable_numbers(x, y):
            return x in self.NumberTexts and y != "-"

        def combinable(i, tag):
            return len(lemmas) > 0 and tag == 'CD' and pos_tags[-1] == 'CD' and \
                   two_combinable_numbers(lemmas[-1], annotation['lemmas'][i])

        tokens = []
        lemmas = []
        pos_tags = []
        ner_tags = []

        for i, tag in enumerate(annotation['pos_tags']):
            if combinable(i, tag) :
                lemmas[-1] = lemmas[-1] + ',' + annotation['lemmas'][i]
                tokens[-1] = tokens[-1] + ',' + annotation['tokens'][i]
                pos_tags[-1] = "CD"
            else:
                lemmas.append(annotation['lemmas'][i])
                tokens.append(annotation['tokens'][i])
                pos_tags.append(annotation['pos_tags'][i])
                ner_tags.append(annotation['ner_tags'][i])

        output = dict(
            tokens=tokens, lemmas=lemmas, pos_tags=pos_tags, ner_tags=ner_tags
        )
        self.assert_equal_length(output)
        return output

    def _tag_url_and_split_number(self, annotation):
        tokens = []
        lemmas = []
        pos_tags = []
        ner_tags = []

        for i, lemma in enumerate(annotation['lemmas']):
            if 'http' in lemma or 'www.' in lemma:
                lemmas.append(lemma)
                tokens.append(annotation['tokens'][i])
                pos_tags.append(annotation['pos_tags'][i])
                ner_tags.append("URL")
            elif re.match(self.DashedNumbers, lemma) and annotation['ner_tags'][i] == 'DATE':
                _lemmas = lemma.replace('-', ' - ').split()
                _tokens = annotation['tokens'][i].replace('-', ' - ').split()
                assert len(_lemmas) == len(_tokens), annotation
                for l in _lemmas:
                    if l != '-':
                        pos_tags.append(annotation['pos_tags'][i])
                        ner_tags.append(annotation['ner_tags'][i])
                    else:
                        pos_tags.append(':')
                        ner_tags.append('0')
                lemmas = lemmas + _lemmas
                tokens = tokens + _tokens
            else:
                lemmas.append(annotation['lemmas'][i])
                tokens.append(annotation['tokens'][i])
                pos_tags.append(annotation['pos_tags'][i])
                ner_tags.append(annotation['ner_tags'][i])

        output = dict(
            tokens=tokens, lemmas=lemmas, pos_tags=pos_tags, ner_tags=ner_tags
        )
        self.assert_equal_length(output)
        return output


# if __name__ == '__main__':
#
#     annotator = FeatureAnnotator()
#
#     for file_path in args.files:
#         # logger.info('Processing {}'.format(file_path))
#         with open(file_path + '.features', 'w', encoding='utf-8') as f:
#             for i, amr in enumerate(AMRIO.read(file_path), 1):
#                 if i % 1000 == 0:
#                     # logger.info('{} processed.'.format(i))
#                 annotation = annotator(amr.sentence)
#                 amr.tokens = annotation['tokens']
#                 amr.lemmas = annotation['lemmas']
#                 amr.pos_tags = annotation['pos_tags']
#                 amr.ner_tags = annotation['ner_tags']
#                 # AMRIO.dump([amr], f)
#     # logger.info('Done!')