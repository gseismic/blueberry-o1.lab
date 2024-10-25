from typing import List, Literal, TypedDict, Sequence, Tuple, Dict, Any
from ...logger import user_logger
from .exception import ParseDialogError

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

# 设计参考:
# https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L222
class ChatFormat:
    def __init__(self, tokenizer, logger=user_logger):
        self.tokenizer = tokenizer
        self.logger = logger

    def encode_role(self, role: Role) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.begin_role_id)
        tokens.extend(self.tokenizer.encode(role, bos=False, eos=False, if_unknown='ignore'))
        tokens.append(self.tokenizer.end_role_id)
        return tokens

    def encode_content(self, content: str) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.begin_content_id)
        tokens.extend(self.tokenizer.encode(content.strip(), bos=False, eos=False, if_unknown='ignore'))
        tokens.append(self.tokenizer.end_content_id)
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = []
        tokens.append(self.tokenizer.begin_message_id)
        tokens.extend(self.encode_role(message["role"]))
        tokens.extend(self.encode_content(message["content"]))
        tokens.append(self.tokenizer.end_message_id)
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog, assistant_role='assistant', return_info=True) -> Tuple[List[int], Dict[str, Any]]:
        tokens = []
        tokens.append(self.tokenizer.begin_text_id)
        message_seps = []
        for message in dialog:
            tokens.extend(self.encode_message(message))
            message_seps.append(len(tokens))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_role(assistant_role))
        if return_info:
            info = {
                'message_seps': message_seps,
                'num_tokens': len(tokens),
            }
            return tokens, info
        else:
            return tokens
    
    def encode_dialog(self, dialog: Dialog, return_info=True) -> Tuple[List[int], Dict[str, Any]]:
        tokens = []
        tokens.append(self.tokenizer.begin_text_id)
        message_seps = []
        for message in dialog:
            tokens.extend(self.encode_message(message))
            message_seps.append(len(tokens))
        tokens.append(self.tokenizer.end_text_id)
        if return_info:
            info = {
                'message_seps': message_seps,
                'num_tokens': len(tokens),
            }
            return tokens, info
        else:
            return tokens

    def decode_dialog(self, tokens: List[int]) -> Dialog:
        assert tokens[0] == self.tokenizer.begin_text_id, f'Dialog should start with begin_text_id={self.tokenizer.begin_text_id}'
        assert tokens[-1] == self.tokenizer.end_text_id, f'Dialog should end with end_text_id={self.tokenizer.end_text_id}, but got {tokens[-1]}={self.tokenizer.decode([tokens[-1]], skip_all_special=False)}'
        messages = []
        while True:
            if tokens[0] == self.tokenizer.end_text_id:
                break
            if tokens[0] == self.tokenizer.begin_role_id:
                tokens.pop(0)  # 移除 begin_role_id
                role_tokens = []
                while tokens[0] != self.tokenizer.end_role_id:
                    role_tokens.append(tokens.pop(0))
                    if len(tokens) == 0:
                        raise ParseDialogError(f'role is not ended by end_role_id={self.tokenizer.end_role_id}')
                tokens.pop(0)  # 移除 end_role_id
                role = self.tokenizer.decode(role_tokens, skip_all_special=False)
                
                # 确保role后面紧跟着content
                assert tokens[0] == self.tokenizer.begin_content_id, f"Each role should be followed by content, but got {tokens[0]}={self.tokenizer.decode([tokens[0]], skip_all_special=False)}"
                tokens.pop(0)  # 移除 begin_content_id
                content_tokens = []
                while True:
                    if tokens[0] == self.tokenizer.end_content_id:
                        tokens.pop(0) # pop end_content_id
                        break
                    content_tokens.append(tokens.pop(0))
                    if len(tokens) == 0:
                        raise ParseDialogError(f'content is not ended by end_content_id={self.tokenizer.end_content_id}')
                content = self.tokenizer.decode(content_tokens, skip_all_special=False)
                
                messages.append(Message(role=role, content=content))
            else:
                tokens.pop(0)
        
        return messages
