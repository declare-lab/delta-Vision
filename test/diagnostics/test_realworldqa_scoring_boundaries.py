from src.benchmarks import score_realworldqa_prediction as score

def test_option_letter_is_not_a_prose_substring():
    c=['Lane change left','Stay in the current lane','Lane change right']
    assert score('John Daly Blvd','B',c)['score']==0
    assert score('The image shows a wet road, not','A',c)['score']==0
    assert score('B. Stay in the current lane','B',c)['score']==1
    assert score('Stay in the current lane','B',c)['score']==1

def test_numbers_are_not_substrings():
    assert score('15','1')['score']==0
    assert score('30','3')['score']==0
    assert score('3500','B',['25','35','45'])['score']==0
    assert score('35','B',['25','35','45'])['score']==1
    assert score('There are 3 cars.','3')['score']==1

def test_short_words_are_not_option_labels():
    assert score('Bike','Bus')['score']==0
    assert score('Bus','Bus')['score']==1
    assert score('blue car','Blue')['score']==1
    assert score('None','No')['score']==0
    assert score('No.','No')['score']==1


def test_option_label_missing_period():
    question="A. first\nB second\nC. third"
    assert score('C','C',question=question)['score']==1
    assert score('second','B',question=question)['score']==1
