"""Explain the existing geometric face comparison and catalog-based choices.

No cranial shape classification, trained accuracy or probability is implied.
Recommendations use the user's reference-sheet face tags and simple priorities.
"""
from salon.catalog import STYLES,FACE_SHAPES

PRIORITIES={
    'oval':[1,19,6], 'round':[5,7,19], 'square':[3,7,16],
    'long':[1,6,15], 'heart':[1,6,8], 'vshape':[4,12,14], 'diamond':[22,2,10],
}
GOALS={
    'oval':'轮廓较均衡，可优先比较刘海露额量、长度与日常打理。',
    'round':'可比较适度顶部高度和侧向发流，观察纵向比例与两侧宽度。',
    'square':'额部、颧部和下颌宽度较接近，可比较侧向发流、适度顶部高度及不同露额量。',
    'long':'可优先比较有刘海、顶部不过高的款式，观察侧部体积与脸长比例。',
    'heart':'可比较分散刘海与自然侧部层次，观察额部和下巴的视觉平衡。',
    'vshape':'可比较有侧部层次和适度后部长度的款式，观察下巴与两侧体积。',
    'diamond':'参考图缺少菱形脸专属标签，可先从标注“所有脸型”的短发中对照。',
}
SQUARE_REASONS={
    3:'后梳和适度顶部高度，便于比较露额后脸部的纵向比例。',
    7:'侧向发流与不对称轮廓，可作为柔化方正观感的选项。',
    16:'短前梳刘海，可与露额款比较额头露出量和轮廓柔和度。',
}


def recommendations(face,styles=STYLES):
    eligible=[style for style in styles if face in style['suitable_faces']]
    priority=PRIORITIES.get(face,[])
    eligible.sort(key=lambda style:(priority.index(style['number']) if style['number'] in priority else 99,style['difficulty'],style['number']))
    result=[]
    for style in eligible[:3]:
        reason=SQUARE_REASONS.get(style['number']) if face=='square' else None
        if not reason:reason='手工款式规则适合'+FACE_SHAPES[face]['name']+'；'+style['intro']
        result.append({'id':style['id'],'number':style['number'],'name':style['name'],'reason':reason,
                       'basis':'manually_authored_face_tags_and_style_features'})
    return result
