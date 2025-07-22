import os
import glob
import xml.etree.ElementTree as ET
import hashlib
from datetime import datetime
from config import NAMESPACES, VERSION
from keyword_extractor import extract_keywords_keybert_fasttext, extract_topics
from reference import extract_core_patent_number, load_reference_rows
import time
import re

# グローバルで1回だけ参照CSVを読み込む
REFERENCE_ROWS = load_reference_rows()

def get_text(element, path, default=None, namespaces=None):
    found = element.find(path, namespaces=namespaces)
    return found.text.strip() if found is not None and found.text else default

def get_all_texts(element, path, namespaces=None):
    return [el.text.strip() for el in element.findall(path, namespaces=namespaces) if el.text]

def get_attr(element, attr, default=None):
    return element.attrib.get(attr, default)

def parse_ipc_entry(ipc_str):
    s = ipc_str.strip()
    # 末尾の空白で分割
    if ' ' in s:
        parts = s.rsplit(' ', 1)
        code = parts[0].replace(' ', '')
        return {
            'code': code,
            'raw': ipc_str.strip()
        }
    # fallback: 空白除去
    return {'code': s.replace(' ', ''), 'raw': ipc_str.strip()}

def parse_patent_xml(xml_path):
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    root = ET.fromstring(xml_content)
    tag = root.tag

    if ('UnexaminedPatentPublication' in tag or
        'RegisteredPatentPublication' in tag or
        'InternationalPatentPublication' in tag):
        return parse_new_format(root, xml_path, xml_content)
    elif tag == 'jp-official-gazette':
        return parse_old_format(root, xml_path, xml_content)
    else:
        raise ValueError(f"Unknown XML format: {tag}")

# 新形式（名前空間付き）
def parse_new_format(root, xml_path, xml_content):
    from config import NAMESPACES, VERSION
    # ルート属性取得
    lang = root.attrib.get('{http://www.wipo.int/standards/XMLSchema/ST96/Common}languageCode', 'ja')
    country = 'JP'  # 固定値でOK
    kind_code = None  # XMLから取得する場合は適宜修正

    # bibliographic-data相当
    tag = root.tag
    if 'UnexaminedPatentPublication' in tag:
        biblio = root.find('.//jppat:UnexaminedPatentPublicationBibliographicData', namespaces=NAMESPACES)
    elif 'RegisteredPatentPublication' in tag:
        biblio = root.find('.//jppat:RegisteredPatentPublicationBibliographicData', namespaces=NAMESPACES)
    elif 'InternationalPatentPublication' in tag:
        biblio = root.find('.//jppat:InternationalPatentPublicationBibliographicData', namespaces=NAMESPACES)
    else:
        raise ValueError(f'Unknown XML format: {tag}')
    if biblio is None:
        raise ValueError(f'biblio not found in {xml_path}')

    patent_id = get_text(biblio, './jppat:PatentPublicationIdentification/pat:PublicationNumber', namespaces=NAMESPACES)
    application_number = get_text(biblio, './jppat:ApplicationIdentification/com:ApplicationNumber/com:ApplicationNumberText', namespaces=NAMESPACES)
    title = get_text(biblio, './pat:InventionTitle', namespaces=NAMESPACES)
    applicants = get_all_texts(biblio, './/com:EntityName', namespaces=NAMESPACES)
    inventors = get_all_texts(biblio, './/jppat:Inventor/jpcom:Contact/com:Name/com:EntityName', namespaces=NAMESPACES)
    filing_date = get_text(biblio, './jppat:ApplicationIdentification/pat:FilingDate', namespaces=NAMESPACES)
    publication_date = get_text(biblio, './jppat:PatentPublicationIdentification/com:PublicationDate', namespaces=NAMESPACES)
    priority_date = None  # 必要なら適宜パスを修正
    classification_national = get_all_texts(biblio, './/jppat:MainNationalClassification/pat:PatentClassificationText', namespaces=NAMESPACES) + get_all_texts(biblio, './/jppat:FurtherNationalClassification/pat:PatentClassificationText', namespaces=NAMESPACES)
    f_term = get_all_texts(biblio, './/jppat:FtermInformation', namespaces=NAMESPACES)
    theme_code = get_all_texts(biblio, './/jppat:ThemeCodeInformation', namespaces=NAMESPACES)
    number_of_claims = get_text(biblio, './/pat:ClaimTotalQuantity', namespaces=NAMESPACES)

    # description
    description_elem = root.find('.//jppat:Description', namespaces=NAMESPACES)
    description = ''
    if description_elem is not None:
        description = ET.tostring(description_elem, encoding='unicode', method='text').strip()

    # claims
    claims = []
    for claim_elem in root.findall('.//pat:Claim', namespaces=NAMESPACES):
        num = claim_elem.attrib.get('com:sequenceNumber', None)
        text = ET.tostring(claim_elem, encoding='unicode', method='text').strip()
        claims.append({'num': num, 'text': text})

    # abstract（このXMLには無い場合もある）
    abstract_elem = root.find('.//pat:Abstract', namespaces=NAMESPACES)
    summary = ''
    if abstract_elem is not None:
        summary = ET.tostring(abstract_elem, encoding='unicode', method='text').strip()

    # IPC分類の構造化
    raw_ipc = get_all_texts(biblio, './/pat:MainClassification', namespaces=NAMESPACES) + get_all_texts(biblio, './/pat:FurtherClassification', namespaces=NAMESPACES)
    classification_ipc = [parse_ipc_entry(ipc) for ipc in raw_ipc]

    # キーワード抽出
    keywords = extract_keywords_keybert_fasttext(summary, topn=10) if summary else []
    topics = extract_topics({'metadata': {'classification_ipc': classification_ipc, 'keywords': keywords}})

    file_size = os.path.getsize(xml_path)
    checksum = hashlib.sha256(xml_content.encode('utf-8')).hexdigest()
    ingest_timestamp = datetime.utcnow().isoformat() + 'Z'

    reference_syutugan = []
    reference_himotsuki = []
    core_patent_id = extract_core_patent_number(patent_id)
    for row in REFERENCE_ROWS:
        if (
            extract_core_patent_number(row['syutugan']) == core_patent_id or
            extract_core_patent_number(row['himotuki']) == core_patent_id
        ):
            reference_syutugan.append(row['syutugan'])
            reference_himotsuki.append(row['himotuki'])
    reference_syutugan = list(set(reference_syutugan))
    reference_himotsuki = list(set(reference_himotsuki))
    reference_flag = bool(reference_syutugan or reference_himotsuki)

    reference_obj = {
        'exist': reference_flag,
        'syutugan': reference_syutugan,
        'himotsuki': reference_himotsuki
    }

    data = {
        'metadata': {
            'patent_id': patent_id,
            'application_number': application_number,
            'title': title,
            'filing_date': filing_date,
            'publication_date': publication_date,
            'reference': reference_obj,
            'classification_ipc': classification_ipc,
            'classification_fi': classification_national,
            'f_term': f_term,
            'theme_code': theme_code,
            'keywords': keywords,
            'topics': topics,
        },
        'source_file': xml_path,
        'ingest_timestamp': ingest_timestamp,
        'parse_version': VERSION,
        'file_size': file_size,
        'applicants': applicants,
        'inventors': inventors,
        'country': country,
        'kind_code': kind_code,
        'language': lang,
        'checksum': checksum,
        'priority_date': priority_date,
        'number_of_claims': number_of_claims,
        'claims': claims,
        'summary': summary,
        'description': description
    }
    return data

# 旧形式（プレーンタグ）
def parse_old_format(root, xml_path, xml_content):
    from config import VERSION, NAMESPACES
    def get_text(element, path, default=None, namespaces=None):
        found = element.find(path, namespaces=namespaces)
        return found.text.strip() if found is not None and found.text else default
    def get_all_texts(element, path, namespaces=None):
        return [el.text.strip() for el in element.findall(path, namespaces=namespaces) if el.text]

    biblio = root.find('bibliographic-data')
    patent_id = get_text(biblio, './publication-reference/document-id/doc-number')
    application_number = get_text(biblio, './application-reference/document-id/doc-number')
    title = get_text(biblio, './invention-title')
    applicants = get_all_texts(biblio, './parties/jp:applicants-agents-article/jp:applicants-agents/applicant/addressbook/name', namespaces=NAMESPACES)
    inventors = get_all_texts(biblio, './parties/inventors/inventor/addressbook/name')
    filing_date = get_text(biblio, './application-reference/document-id/date')
    publication_date = get_text(biblio, './publication-reference/document-id/date')
    priority_date = get_text(biblio, './priority-claims/priority-claim/date')
    classification_national = get_all_texts(biblio, './classification-national/main-clsf') + get_all_texts(biblio, './classification-national/further-clsf')
    f_term = get_all_texts(biblio, './/jp:f-term', namespaces=NAMESPACES)
    theme_code = get_all_texts(biblio, './/jp:theme-code', namespaces=NAMESPACES)
    number_of_claims = get_text(biblio, './number-of-claims')

    description_elem = root.find('description')
    description = ''
    if description_elem is not None:
        description = ET.tostring(description_elem, encoding='unicode', method='text').strip()

    claims = []
    for claim_elem in root.findall('.//claim'):
        num = claim_elem.attrib.get('num', None)
        claim_text_elem = claim_elem.find('claim-text')
        text = ''
        if claim_text_elem is not None:
            text = ET.tostring(claim_text_elem, encoding='unicode', method='text').strip()
        claims.append({'num': num, 'text': text})

    abstract_elem = root.find('abstract')
    summary = ''
    if abstract_elem is not None:
        summary = ET.tostring(abstract_elem, encoding='unicode', method='text').strip()

    # IPC分類の構造化
    raw_ipc = get_all_texts(biblio, './classification-ipc/main-clsf') + get_all_texts(biblio, './classification-ipc/further-clsf')
    classification_ipc = [parse_ipc_entry(ipc) for ipc in raw_ipc]

    # キーワード抽出
    keywords = extract_keywords_keybert_fasttext(summary, topn=10) if summary else []
    topics = extract_topics({'metadata': {'classification_ipc': classification_ipc, 'keywords': keywords}})

    file_size = os.path.getsize(xml_path)
    checksum = hashlib.sha256(xml_content.encode('utf-8')).hexdigest()
    ingest_timestamp = datetime.utcnow().isoformat() + 'Z'

    reference_syutugan = []
    reference_himotsuki = []
    core_patent_id = extract_core_patent_number(patent_id)
    for row in REFERENCE_ROWS:
        if (
            extract_core_patent_number(row['syutugan']) == core_patent_id or
            extract_core_patent_number(row['himotuki']) == core_patent_id
        ):
            reference_syutugan.append(row['syutugan'])
            reference_himotsuki.append(row['himotuki'])
    reference_syutugan = list(set(reference_syutugan))
    reference_himotsuki = list(set(reference_himotsuki))
    reference_flag = bool(reference_syutugan or reference_himotsuki)

    reference_obj = {
        'exist': reference_flag,
        'syutugan': reference_syutugan,
        'himotsuki': reference_himotsuki
    }

    data = {
        'metadata': {
            'patent_id': patent_id,
            'application_number': application_number,
            'title': title,
            'filing_date': filing_date,
            'publication_date': publication_date,
            'reference': reference_obj,
            'classification_ipc': classification_ipc,
            'classification_fi': classification_national,
            'f_term': f_term,
            'theme_code': theme_code,
            'keywords': keywords,
            'topics': topics,
        },
        'source_file': xml_path,
        'ingest_timestamp': ingest_timestamp,
        'parse_version': VERSION,
        'file_size': file_size,
        'applicants': applicants,
        'inventors': inventors,
        'country': 'JP',
        'kind_code': None,
        'language': 'ja',
        'checksum': checksum,
        'priority_date': priority_date,
        'number_of_claims': number_of_claims,
        'claims': claims,
        'summary': summary,
        'description': description
    }
    return data

def find_xml_file_by_patent_code(patent_code):
    pattern = os.path.join('input_files', '*', '*', patent_code, 'text.txt')
    files = glob.glob(pattern)
    return files[0] if files else None 