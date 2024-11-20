import re
from lxml import etree
from llm_data.api.base import BaseAPI
from llm_data.utils.parser.html import space_like_removed


class DictCnAPI(BaseAPI):
    _BASE_URL = 'https://dict.cn/'
    _BASE_HY_URL = 'https://hanyu.dict.cn/'

    def __init__(self, *args, **kwargs):
        super(DictCnAPI, self).__init__(*args, **kwargs)
    
    def _parse_abstract(self, section):
        # pinyin: //*[@id="cy"]/div[1]/div/p/span[1]
        out_data = {}
        py = section.xpath('./p[contains(@class, "hz_pinyin")]')
        assert len(py) == 1
        # assert len(py_spans) == 2
        assert space_like_removed(py[0].xpath('./span[@class="zh_title"]/text()')[0]) == '拼音'
        pinyins = [space_like_removed(item) for item in py[0].xpath('./span[not(@class)]/text()')]

        # out_data['拼音'] = pinyin
        # pinyin = section.xpath('./p[contains(@class, "hz_pinyin")]/span[@class="zh_title"]/following-sibling::span[1]/text()')[0]
        out_data['拼音'] = pinyins
        _uls = section.xpath('./ul')
        assert len(_uls) == 3
        
        _ul1_li = _uls[0].xpath('./li')
        assert len(_ul1_li) == 2
        for i, label in enumerate(['繁体', '异体']):
            assert space_like_removed(_ul1_li[i].xpath('./span[1]/text()')[0]) == label
            values = _ul1_li[i].xpath('./span[2]/text()')
            if not values:
                values = _ul1_li[i].xpath('./strong/a/text()')
                # if values:
                #     yiti_href = space_like_removed(_ul1_li[i].xpath('./strong/a/@href')[0])
            value = None if len(values) == 0 else space_like_removed(values[0])
            out_data[label] = value

        _ul2_li = _uls[1].xpath('./li')
        assert len(_ul2_li) == 3
        for i, label in enumerate(['部首', '笔画', '五笔']):
            assert space_like_removed(_ul2_li[i].xpath('./span[1]/text()')[0]) == label
            values = _ul2_li[i].xpath('./span[2]/text()')
            out_data[label] =  None if len(values) == 0 else space_like_removed(values[0])
            
        _ul3_li = _uls[2].xpath('./li')
        assert len(_ul3_li) == 2
        for i, label in enumerate(['结构', '造字法']):
            assert space_like_removed(_ul3_li[i].xpath('./span[1]/text()')[0]) == label
            values = _ul3_li[i].xpath('./span[2]/text()')
            out_data[label] =  None if len(values) == 0 else space_like_removed(values[0])

        return out_data

    def _parse_cixin(self, tag):
        # 1.\xa0[代]
        assert '\xa0' in tag
        tags = re.findall(r'\[(.*?)\]', tag)
        assert len(tags) == 1
        return tags[0]
    
    def _parse_meaning_section(self, sections, word):   
        """
        现代汉语:
        <div[@id='ly']/ul/li>
            <h2>读音1
            <ul>[该读音下词组列表]
                <div/em/span>具体词组 此条目~代表的含义,e.g. `什么`，如果没有，则代表汉字本身
                <li>
                    - <em>名词
                    - <p/span>释意1, <p/span/em>['~xx', 'y~']
                    - <p/span>释意2, <p/span/em>['~xx', 'y~']
                <li>
                    - <em>名词
                    - <p/span>释意1, <p/span/em>['~xx', 'y~']
                    - <p/span>释意2, <p/span/em>['~xx', 'y~']
        <li>读音2
        """
        out_data = []
        for i, para in enumerate(sections):
            # <li style='padding:0;>
            # 每个读音的词组信息
            data = {}
            parsed_word = space_like_removed(para.xpath('./h2/text()')[0])
            assert parsed_word == word
            pinyin = space_like_removed(para.xpath('./h2/span/text()')[0])
            data['pinyin'] = pinyin
            data['ciyu_list'] = []
            
            # my_data['ciyu'] = 
            _ul_list = para.xpath('./ul[not(@style)]')
            # _li_list = para.xpath('./ul[not(@style)]/li') 
            assert len(_ul_list) >= 1
            for _ul in _ul_list:
                # 遍历每个[词语]
                ciyu_info = {}
                ciyu = _ul.xpath('./div/span/text()')
                if not ciyu:
                    ciyu_info['ciyu'] = None # 没有词组时，~代表word本身
                    ciyu_info['ciyu_pinyin'] = None
                else:
                    assert len(ciyu) == 1
                    __tu = ciyu[0].split('\xa0')
                    ciyu_info['ciyu'] = space_like_removed(__tu[0])
                    ciyu_info['ciyu_pinyin'] = space_like_removed(__tu[1])
            
                _li_list = _ul.xpath('./li')                
                meaning_info = [] # 该词所有词性的所有意思
                for _li in _li_list:
                    # 遍历每个【词性】
                    meaning_type, meaning_list = None, []
                    _type = _li.xpath('./em/text()')
                    assert len(_type) == 1
                    meaning_type = self._parse_cixin(_type[0])
                    _li_p = _li.xpath('./p') # meaning list
                    # assert len(_li_p) >= 1 # https://hanyu.dict.cn/%E8%93%9D 
                    meaning_list = []
                    for item in _li_p:
                        _li_p_span = item.xpath('./span')
                        assert len(_li_p_span) == 1
                        meaning = space_like_removed(_li_p_span[0].xpath('./text()')[0])
                        words = _li_p_span[0].xpath('./em[@class="hz_sy"]/text()')
                        assert len(words) in [0, 1]
                        if words:
                            # assert '|' not in words[0]
                            words = [w.strip() for w in words[0].replace('|', '｜').split('｜')]
                        meaning_list.append({'meaning': meaning,'words': words})
                    meaning_info.append({'meaning_type': meaning_type, 'meaning_list': meaning_list})
                
                ciyu_info['meaning_info'] = meaning_info
                data['ciyu_list'].append(ciyu_info)
            out_data.append(data)
        
        return out_data
    
    def parse_hy_word(self, word, text):
        tree = etree.HTML(text)
        _word_parsed = tree.xpath('//*[@id="ly"]/ul/li/h2/text()')
        if len(_word_parsed) == 0:
            return None
        word_parsed = _word_parsed[0].strip()
        assert word_parsed == word
        # [名]: //*[@id="ly"]/ul/li/ul[2]/li/em
        # 私心: //*[@id="ly"]/ul/li/ul[2]/li/p/span/text()
        # 大公无～: //*[@id="ly"]/ul/li/ul[2]/li/p/span/em
        # item = tree.xpath('//*[@id="ly"]/ul/li/ul[1]/li/p[1]/span[1]')[0]
        abstract_section = tree.xpath('//*[@id="cy"]/div[1]/div')
        assert len(abstract_section) == 1
        abstract_info = self._parse_abstract(abstract_section[0])
        sections = tree.xpath('//*[@id="ly"]/ul/li')
        # assert len(section) == 1
        meaning_info = self._parse_meaning_section(sections, word=word)
        out_data = {
            'abstract': abstract_info,
            'meanings': meaning_info,
        }
        return out_data
    
    def get_hy_word_html(self, word):
        url = self._BASE_HY_URL + word
        return self.spider.get_source(url)
    
    def get_hy_word(self, word):
        text = self.get_hy_word_html(word)
        return self.parse_hy_word(word, text)
