from typing import List


def rows_to_chat(rows: List[List[dict]]) -> List[dict]:
    """
    Converts a list of rows to a list of chats. Each row is a list of messages.

    We assume that the following columns exist in the rows:
    - 'system_prompt'
    - 'one_shot_user_input'
    - 'one_shot_assistant_output'
    - 'user_input'
    """

    chats = []
    for system_prompt, one_shot_user_input, one_shot_assistant_output, user_input in \
        zip(rows['system_prompt'], rows['one_shot_user_input'], rows['one_shot_assistant_output'], rows['user_input']):
        chats.append([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': one_shot_user_input},
            {'role': 'assistant', 'content': one_shot_assistant_output},
            {'role': 'user', 'content': user_input}
        ])

    return chats


def rows_to_chat_multi(rows: List[List[dict]]) -> List[dict]:
    """
    Converts a list of rows to a list of chats with multiple few-shot examples.
    
    Assumes the following columns exist in the rows:
    - 'system_prompt': System instruction
    - 'few_shot_messages': Pre-formatted string of alternating user/assistant examples
    - 'user_input': The final user query
    
    Example few_shot_messages format:
    "User: CONTEXT: Some context
    STATEMENT: Example statement 1
    Assistant: YES
    User: CONTEXT: Another context
    STATEMENT: Example statement 2
    Assistant: NO
    "
    """
    chats = []
    for system_prompt, few_shot_messages, user_input in \
        zip(rows['system_prompt'], rows['few_shot_messages'], rows['user_input']):
        
        messages = [{'role': 'system', 'content': system_prompt}]
        
        # Parse few-shot messages into individual user/assistant pairs
        if few_shot_messages and few_shot_messages.strip():
            lines = few_shot_messages.strip().split('\n')
            i = 0
            while i < len(lines):
                line = lines[i]
                
                if line.startswith('User: '):
                    # Collect all content until we hit 'Assistant:'
                    user_content = [line.replace('User: ', '', 1)]
                    i += 1
                    while i < len(lines) and not lines[i].startswith('Assistant: '):
                        user_content.append(lines[i])
                        i += 1
                    
                    messages.append({
                        'role': 'user',
                        'content': '\n'.join(user_content)
                    })
                    
                    # Now get the Assistant response
                    if i < len(lines) and lines[i].startswith('Assistant: '):
                        assistant_content = [lines[i].replace('Assistant: ', '', 1)]
                        i += 1
                        while i < len(lines) and not lines[i].startswith('User: '):
                            assistant_content.append(lines[i])
                            i += 1
                        
                        messages.append({
                            'role': 'assistant',
                            'content': '\n'.join(assistant_content)
                        })
                else:
                    i += 1
        
        # Add the final user input
        messages.append({'role': 'user', 'content': user_input})
        chats.append(messages)

    return chats
