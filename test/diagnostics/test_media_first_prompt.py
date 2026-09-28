"""Optional legacy renderer tests without importing Torch or model weights."""
import ast
from pathlib import Path
import re
import unittest


def render():
    source = Path(__file__).resolve().parents[3] / 'vision-kv-inject-attention-sink/src/data.py'
    if not source.is_file():
        raise unittest.SkipTest('Optional legacy renderer checkout is not available')
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'media_first_qwen_content')
    namespace = {'re': re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[function.name]


def test_references_do_not_reorder_or_duplicate_pixels():
    renderer = render()
    images = [object(), object()]
    question = 'Compare <|image_2|> to <|image_1|>.\nA. <|image_2|>\nB. none'
    result = renderer(question, images, [])
    assert [x['image'] for x in result if x['type'] == 'image'] == images
    assert result[-1] == {'type': 'text', 'text': 'Compare Image 2 to Image 1.\nA. Image 2\nB. none'}
    assert result[1]['text'] == '\n[End of Image 1]\n'
    assert result[3]['text'] == '\n[End of Image 2]\n'


def test_no_marker_multiimage_input_is_unchanged():
    renderer = render()
    images = [object(), object()]
    assert renderer('Question and options', images, []) == [
        {'type': 'image', 'image': image} for image in images
    ] + [{'type': 'text', 'text': 'Question and options'}]


def test_video_question_follows_video_without_other_changes():
    renderer = render()
    video = object()
    assert renderer('Question', [], [video]) == [
        {'type': 'video', 'video': video}, {'type': 'text', 'text': 'Question'}]


def test_out_of_range_reference_is_rejected():
    renderer = render()
    for reference in ['<|image_0|>', '<|image_3|>', '<|video_1|>']:
        try:
            renderer(reference, [object(), object()], [])
        except ValueError:
            continue
        raise AssertionError(f'Invalid reference accepted: {reference}')


if __name__ == '__main__':
    tests = [value for name, value in globals().copy().items() if name.startswith('test_')]
    result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in tests))
    raise SystemExit(not result.wasSuccessful())
