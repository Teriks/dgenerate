import unittest

import dgenerate.textprocessing as _tp


class TestConceptUriParser(unittest.TestCase):

    def test_known_args(self):
        p = _tp.ConceptUriParser('test', ['arg1', 'arg2', 'arg3'])

        # invalid args

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;arg2')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;arg4')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;arg4=5')

        # empty

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('   ')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('  ; ')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse(';')

        # stray semicolon
        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;arg1=1;')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;arg1=1;arg2=2;')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;arg1=1;arg2=2;arg3=3;')

        # 2 stray semicolon
        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;arg1=1;;')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;arg1=1;arg2=2;;')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;arg1=1;arg2=2;arg3=3;;')

        # 3 stray semicolon
        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;arg1=1;;;')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;arg1=1;arg2=2;;;')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept;arg1=1;arg2=2;arg3=3;;;')

        self.assertDictEqual(p.parse('concept').args, dict())

        # all args

        prs = p.parse('concept1;arg1=1;arg2=true;arg3="h;i"')
        self.assertEqual(prs.concept, 'concept1')
        self.assertDictEqual(prs.args, {'arg1': '1', 'arg2': 'true', 'arg3': 'h;i'})

        prs = p.parse('concept2;arg1="1;";arg2=false;arg3=ho')
        self.assertEqual(prs.concept, 'concept2')
        self.assertDictEqual(prs.args, {'arg1': '1;', 'arg2': 'false', 'arg3': 'ho'})

        prs = p.parse('concept3;arg1=1;arg2=true;arg3=";hi"')
        self.assertEqual(prs.concept, 'concept3')
        self.assertDictEqual(prs.args, {'arg1': '1', 'arg2': 'true', 'arg3': ';hi'})

        prs = p.parse('concept4;arg1=";1";arg2=false;arg3=ho')
        self.assertEqual(prs.concept, 'concept4')
        self.assertDictEqual(prs.args, {'arg1': ';1', 'arg2': 'false', 'arg3': 'ho'})

        prs = p.parse("concept5;arg1=1;arg2=true;arg3=';hi'")
        self.assertEqual(prs.concept, 'concept5')
        self.assertDictEqual(prs.args, {'arg1': '1', 'arg2': 'true', 'arg3': ';hi'})

        prs = p.parse("concept6;arg1='1;';arg2=false;arg3=ho")
        self.assertEqual(prs.concept, 'concept6')
        self.assertDictEqual(prs.args, {'arg1': '1;', 'arg2': 'false', 'arg3': 'ho'})

        prs = p.parse("concept7;arg1=1;arg2=true;arg3='h;i'")
        self.assertEqual(prs.concept, 'concept7')
        self.assertDictEqual(prs.args, {'arg1': '1', 'arg2': 'true', 'arg3': 'h;i'})

        prs = p.parse("concept8;arg1='1;';arg2=false;arg3=ho")
        self.assertEqual(prs.concept, 'concept8')
        self.assertDictEqual(prs.args, {'arg1': '1;', 'arg2': 'false', 'arg3': 'ho'})

        # some args

        prs = p.parse('concept1;arg1=1;arg3=";hi"')
        self.assertEqual(prs.concept, 'concept1')
        self.assertDictEqual(prs.args, {'arg1': '1', 'arg3': ';hi'})

        prs = p.parse('concept2;arg1=";1";arg2=false')
        self.assertEqual(prs.concept, 'concept2')
        self.assertDictEqual(prs.args, {'arg1': ';1', 'arg2': 'false'})

        prs = p.parse('concept3;arg2=true;arg3=";hi"')
        self.assertEqual(prs.concept, 'concept3')
        self.assertDictEqual(prs.args, {'arg2': 'true', 'arg3': ';hi'})

        prs = p.parse('concept4;arg1="1;";arg2=false')
        self.assertEqual(prs.concept, 'concept4')
        self.assertDictEqual(prs.args, {'arg1': '1;', 'arg2': 'false'})

        prs = p.parse("concept5;arg1=1;arg3='h;i'")
        self.assertEqual(prs.concept, 'concept5')
        self.assertDictEqual(prs.args, {'arg1': '1', 'arg3': 'h;i'})

        prs = p.parse("concept6;arg1='1;';arg2=false")
        self.assertEqual(prs.concept, 'concept6')
        self.assertDictEqual(prs.args, {'arg1': '1;', 'arg2': 'false'})

        prs = p.parse("concept7;arg2=true;arg3='hi;'")
        self.assertEqual(prs.concept, 'concept7')
        self.assertDictEqual(prs.args, {'arg2': 'true', 'arg3': 'hi;'})

        prs = p.parse("concept8;arg1='1;';arg2=false")
        self.assertEqual(prs.concept, 'concept8')
        self.assertDictEqual(prs.args, {'arg1': '1;', 'arg2': 'false'})

    def test_multiple_args(self):
        p = _tp.ConceptUriParser('test',
                                 known_args=['arg1', 'arg2', 'arg3'],
                                 args_lists=['arg2'])

        prs = p.parse('  conc,ept1  ;arg1="hell;o";arg2=",hi", !  ,\'this,\',"sho,uld","p,a,r,s,e"')

        # token is stripped
        self.assertEqual(prs.concept, 'conc,ept1')

        self.assertEqual(prs.args.get('arg1'), 'hell;o')

        self.assertListEqual(prs.args.get('arg2'), [',hi', '!', 'this,', 'sho,uld', 'p,a,r,s,e'])

        prs = p.parse('concept2;arg1=  not,a,list;arg2= a ,  list  ,  , ')

        self.assertEqual(prs.concept, 'concept2')

        self.assertEqual(prs.args.get('arg1'), 'not,a,list')

        self.assertListEqual(prs.args.get('arg2'), ['a', 'list', '', ''])

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept3;arg2="cant"  intermix, \'strings\'')

        with self.assertRaises(_tp.ConceptUriParseError):
            p.parse('concept3;arg1="cant"  intermix \'strings\'')
