FOR_GROUNDING = """
Please describe the objects in the image and the groups they form.  For objects, give the category and a short description. For groups, organize them around functional cooperation and interaction between objects.

The output format should be a Python dictionary, as shown in the example below:

{
    "scene_summary": "The image shows a white flat-panel TV with a black base placed on a brown rectangular wooden side table, and a wooden bed",
    "objects": [
        {
            "instance_id": 1,
            "category": "side table",
            "description": "a brown rectangular wooden side table with a smooth surface, about 60cm in length."
        },
        {
            "instance_id": 2,
            "category": "TV",
            "description": "a white flat-panel TV with a black plastic base, screen size about 55 inches."
        },
        {
            "instance_id": 3,
            "category": "bed",
            "description": "a wooden bed with a black bedding and a white pillow."
        },
    ],
    "groups": [
        {
            "instances": [1, 2],
            "description": "A group consists of a side table and a TV, for viewing TV content and entertainment."
        }
    ]
}

or

{
    "scene_summary": "The image only includes some wall, door and floor, nothing important.",
    "objects": [],
    "groups": []
}


The other rules are as follows:
1. Due to the limitations of subsequent object detectors, each object category shall appear only once.
2. Each group must consist of two or more objects. If there is no group that meets the criteria, the corresponding position shall be left as an empty list.
3. Routine objects with little semantic value shall be ignored, such as doors, door frames, mirrors, walls, floors, etc.
4. Do not split an integral object into parts. For example, do not split "bed" into excessively trivial phrases such as "bed frame" and "bedding"; place detailed information in the "description" field.
"""

FOR_SAME_MAP_COORD_INSTANCE_SIMILARITY = """
Two object detectors have found instances that overlap on a 2D map. Please decide whether the two instances are essentially describing the same object and whether they should be merged. Base your decision on the following information:
- Instance 1 type: {instance1_type}
- Instance 1 description: {instance1_description}
- Instance 2 type: {instance2_type}
- Instance 2 description: {instance2_description}

Judging criteria:
1. If the types of the two instances are similar and their descriptive features are consistent, they are considered the same object.
2. If there is a merge-eligible relationship between the instances—especially a part-to-whole relationship—then they may be merged.

Your output must be a Python dictionary in one of the following forms:

{{
    "reasoning": "Instance 1 and Instance 2 are both tables. Since they overlap when projected onto the 2D map and their descriptions both state they are white, they refer to the same object.",
    "should_merge": True,
}}

or

{{
    "reasoning": "Instance 1 is a table and Instance 2 is a chair. Even if they overlap when projected onto the 2D map, they should not be merged.",
    "should_merge": False,
}}

or

{{
    "reasoning": "Instance 1 is a bed and Instance 2 is a bedsheet. Given that they overlap on the 2D map, they should be merged into a single bed instance, incorporating the bedsheet's information into the bed's description.",
    "should_merge": True,
}}
"""

FOR_MERGE_INSTANCE_CATEGORY_DESCRIPTION = """
Merge the types and descriptions of two similar instances into a single instance.

Instance 1 category: {category_1}
Instance 1 description: {description_1}
Instance 2 category: {category_2}
Instance 2 description: {description_2}

The output format should be a Python dictionary, as shown in the example below:

{{
    "merged_category": "merged category in short",
    "merged_description": "merged description"
}}
"""


FOR_RENEW_INSTANCE_CATEGORY_DESCRIPTION = """
For an object, due to previous observations from different angles and detection errors, the type and description of this object may be inconsistent. Please provide a summarized type and description of the object based on the previous observation results.

The historical observation information is as follows:
{history_categories_descriptions}

You need to combine the reliable information and provide the type and description of this object. Note that sometimes different parts of an object might be considered as multiple different categories, and you should also summarize them into one object type and description.

Your output format should be a Python dictionary, as shown in the example below:
{{
    "category": "New category, short and representative",
    "description": "New description, summarizing various valid information from before"
}}
"""

FOR_GENERATE_GROUP_DESCRIPTION = """
Based on the descriptions of the following instances, generate a group description. The input information is as follows:{instance_sentences}

The group description should be a concise and representative phrase that can describe the common characteristics of all instances in the group.

Your output format should be a string, as shown in the example below:
{{
    "group_description": "A group consists of a table and a chair, used for rest and entertainment."
}}
"""

FOR_INSTANCE_TARGET_VALUE = """
You are in a home environment looking for {target}.

You see an object: {instance_sentence}

What is the probability that the target is near this object? Please return a value between 0 and 1. Your output format should be a string, as shown in the example below:

{{
    "answer": value
}}
"""

FOR_GROUP_TARGET_VALUE = """
You are in a home environment looking for {target}.

You see a group of objects: {group_description}

What is the probability that the target is near this group? Please return a value between 0 and 1. Your output format should be a string, as shown in the example below:

{{
    "answer": value
}}
"""
