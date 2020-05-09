import torch
from torch import nn
from torch.nn import functional as F

from transformers import BertModel, BertPreTrainedModel
# from transformers import add_start_docstrings, add_start_docstrings_to_callable

from loss import FocalLoss, DSCLoss, DiceLoss, LabelSmoothingCrossEntropy
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss

# @add_start_docstrings(
#     """Bert Model with a token classification head on top (a linear layer on top of
#     the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
#     BERT_START_DOCSTRING,
# )
class BertForTokenClassificationWithDiceLoss(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss()
            loss_fct = DiceLoss()
            # loss_fct = DSCLoss()
            # loss_fct= LabelSmoothingCrossEntropy()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                # print(active_loss, active_loss.shape, \
                #      active_logits,active_logits.shape,\
                #      active_labels,active_labels.shape,\
                #      labels, labels.shape)
                #2048 2048*435 2048 8*256 
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertForTokenClassificationWithTrigger(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            token_type_ids= torch.zeros_like(token_type_ids),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # add triggrt embedding
        for i in range(sequence_output.size(0)):
            trigger_output=[]
            for j in range(sequence_output.size(1)):
                if token_type_ids[i][j]:
                    trigger_output.append(sequence_output[i][j])
            trigger_output = torch.stack(trigger_output,dim=0)
            trigger_output = torch.mean(trigger_output,dim=0)
            sequence_output[i] += trigger_output

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            
            loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss()
            # loss_fct = DiceLoss()
            # loss_fct = DSCLoss()
            # loss_fct= LabelSmoothingCrossEntropy()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                # print(active_loss, active_loss.shape, \
                #      active_logits,active_logits.shape,\
                #      active_labels,active_labels.shape,\
                #      labels, labels.shape)
                #2048 2048*435 2048 8*256 
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

class BertForTokenBinaryClassificationJoint(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.trigger_num_labels = config.trigger_num_labels
        self.role_num_labels = config.role_num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.trigger_start_classifier = nn.Linear(config.hidden_size, config.trigger_num_labels)
        self.trigger_end_classifier = nn.Linear(config.hidden_size, config.trigger_num_labels)
        self.role_start_classifier = nn.Linear(config.hidden_size, config.role_num_labels)
        self.role_end_classifier = nn.Linear(config.hidden_size, config.role_num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        trigger_start_labels=None, # batch * trigger_num_class * seq_length 
        trigger_end_labels=None,
        role_start_labels=None, # batch* role_num_class * seq_length
        role_end_labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        outputs= outputs[2:]

        #######################################################
        ## trigger
        sequence_output_trigger = self.dropout(sequence_output)
        trigger_start_logits = self.trigger_start_classifier(sequence_output_trigger)
        trigger_end_logits = self.trigger_end_classifier(sequence_output_trigger)

        if trigger_start_labels is not None and trigger_end_labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss(class_num=self.trigger_num_labels)
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_trigger_start_logits = trigger_start_logits.view(-1, self.trigger_num_labels)
                active_trigger_end_logits = trigger_end_logits.view(-1, self.trigger_num_labels)

                active_trigger_start_labels = trigger_start_labels.view(-1, self.trigger_num_labels)
                active_trigger_end_labels = trigger_end_labels.view(-1, self.trigger_num_labels)
                # attention_mask: 
                # ignore_index: [cls], [sep]
                # non_index: no label

                # print(active_loss, active_loss.shape, \
                #      active_logits,active_logits.shape,\
                #      active_labels,active_labels.shape,\
                #      labels, labels.shape)
                #2048 2048*435 2048 8*256 
                trigger_start_loss = loss_fct(active_trigger_start_logits, active_trigger_start_labels.float())
                trigger_start_loss = trigger_start_loss * (active_loss.unsqueeze(-1))
                trigger_start_loss = torch.sum(trigger_start_loss)/torch.sum(active_loss)

                trigger_end_loss = loss_fct(active_trigger_end_logits, active_trigger_end_labels.float())
                trigger_end_loss = trigger_end_loss * (active_loss.unsqueeze(-1))
                trigger_end_loss = torch.sum(trigger_end_loss)/torch.sum(active_loss)

            else:
                trigger_start_loss = loss_fct(trigger_start_logits.view(-1, self.trigger_num_labels), trigger_start_labels.view(-1))
                trigger_end_loss = loss_fct(trigger_end_logits.view(-1, self.trigger_num_labels), trigger_end_labels.view(-1))
            trigger_loss = trigger_start_loss+ trigger_end_loss


        #######################################################
        ## role
        # add triggrt embedding
        for i in range(sequence_output.size(0)):
            trigger_output=[]
            for j in range(sequence_output.size(1)):
                if token_type_ids[i][j]:
                    trigger_output.append(sequence_output[i][j])
            trigger_output = torch.stack(trigger_output,dim=0)
            trigger_output = torch.mean(trigger_output,dim=0)
            sequence_output[i] += trigger_output

        sequence_output_role = self.dropout(sequence_output)
        role_start_logits = self.role_start_classifier(sequence_output_role)
        role_end_logits = self.role_end_classifier(sequence_output_role)

        if role_start_labels is not None and role_end_labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss(class_num=self.role_num_labels)
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_role_start_logits = role_start_logits.view(-1, self.role_num_labels)
                active_role_end_logits = role_end_logits.view(-1, self.role_num_labels)

                active_role_start_labels = role_start_labels.view(-1, self.role_num_labels)
                active_role_end_labels = role_end_labels.view(-1, self.role_num_labels)
                # attention_mask: 
                # ignore_index: [cls], [sep]
                # non_index: no label

                # print(active_loss, active_loss.shape, \
                #      active_logits,active_logits.shape,\
                #      active_labels,active_labels.shape,\
                #      labels, labels.shape)
                #2048 2048*435 2048 8*256 
                role_start_loss = loss_fct(active_role_start_logits, active_role_start_labels.float())
                role_start_loss = role_start_loss * (active_loss.unsqueeze(-1))
                role_start_loss = torch.sum(role_start_loss)/torch.sum(active_loss)

                role_end_loss = loss_fct(active_role_end_logits, active_role_end_labels.float())
                role_end_loss = role_end_loss * (active_loss.unsqueeze(-1))
                role_end_loss = torch.sum(role_end_loss)/torch.sum(active_loss)

            else:
                role_start_loss = loss_fct(role_start_logits.view(-1, self.role_num_labels), role_start_labels.view(-1))
                role_end_loss = loss_fct(role_end_logits.view(-1, self.role_num_labels), role_end_labels.view(-1))
            role_loss = role_start_loss+ role_end_loss
            
        outputs = (trigger_loss+ role_loss,) 

        return outputs  # (loss), scores, (hidden_states), (attentions)

  # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def predict(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        trigger_start_labels=None, # batch * trigger_num_class * seq_length 
        trigger_end_labels=None,
        role_start_labels=None, # batch* role_num_class * seq_length
        role_end_labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        outputs= outputs[2:]

        #######################################################
        ## trigger
        sequence_output_trigger = self.dropout(sequence_output)
        trigger_start_logits = self.trigger_start_classifier(sequence_output_trigger)
        trigger_end_logits = self.trigger_end_classifier(sequence_output_trigger)

        # # Only keep active parts 
        # if attention_mask is not None: # 64*256
        #     active_trigger_start_logits  = trigger_start_logits * (attention_mask.unsqueeze(-1))
        #     active_trigger_end_logits = trigger_end_logits * (attention_mask.unsqueeze(-1))
        
        threshold = 0.5
    
        start_preds = torch.sigmoid(start_preds)> threshold # 1498*256*217
        end_preds = torch.sigmoid(end_preds) > threshold

        # 64*256*65
        batch_size, seq_length, num_labels=  trigger_start_logits.size()

        batch_trigger_list = []
        dis = 160
        # trigger
        for i in range(batch_size):   # batch_index
            cur_trigger_list=[]
            for j in range(seq_length):  # token_index 
                if not attention_mask[i, j]: continue
                # 实体 头
                for k in range(num_labels):  
                    if trigger_start_logits[i][j][k]:
                        # 寻找 实体尾 
                        for l in range(j, min(j+ dis, seq_length)):
                            if trigger_end_logits[i][l][k]:
                                cur_trigger_list.append((i, j, l, k)) # batch, start, end, label
                                break
            batch_trigger_list.append(cur_trigger_list)

            
            #######################################################
            ## role
            # add triggrt embedding
            for i in range(sequence_output.size(0)):
                trigger_output=[]
                for j in range(sequence_output.size(1)):
                    if token_type_ids[i][j]:
                        trigger_output.append(sequence_output[i][j])
                trigger_output = torch.stack(trigger_output,dim=0)
                trigger_output = torch.mean(trigger_output,dim=0)
                sequence_output[i] += trigger_output

            sequence_output_role = self.dropout(sequence_output)
            role_start_logits = self.role_start_classifier(sequence_output_role)
            role_end_logits = self.role_end_classifier(sequence_output_role)

            if role_start_labels is not None and role_end_labels is not None:
                # loss_fct = CrossEntropyLoss()
                # loss_fct = FocalLoss(class_num=self.role_num_labels)
                loss_fct = BCEWithLogitsLoss(reduction="none")
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_role_start_logits = role_start_logits.view(-1, self.role_num_labels)
                    active_role_end_logits = role_end_logits.view(-1, self.role_num_labels)



            outputs = (trigger_loss+ role_loss,) 

            return outputs  # (loss), scores, (hidden_states), (attentions)


class BertForTokenBinaryClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.end_classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_labels=None, # batch* num_class* seq_length
        end_labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        start_logits = self.start_classifier(sequence_output)
        end_logits = self.end_classifier(sequence_output)

        outputs = ([start_logits, end_logits],) + outputs[2:]  # add hidden states and attention if they are here
        if start_labels is not None and end_labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss(class_num=self.num_labels)
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_start_logits = start_logits.view(-1, self.num_labels)
                active_end_logits = end_logits.view(-1, self.num_labels)

                active_start_labels = start_labels.view(-1, self.num_labels)
                active_end_labels = end_labels.view(-1, self.num_labels)
                # attention_mask: 
                # ignore_index: [cls], [sep]
                # non_index: no label

                # print(active_loss, active_loss.shape, \
                #      active_logits,active_logits.shape,\
                #      active_labels,active_labels.shape,\
                #      labels, labels.shape)
                #2048 2048*435 2048 8*256 
                start_loss = loss_fct(active_start_logits, active_start_labels.float())
                start_loss = start_loss * (active_loss.unsqueeze(-1))
                start_loss = torch.sum(start_loss)/torch.sum(active_loss)

                end_loss = loss_fct(active_end_logits, active_end_labels.float())
                end_loss = end_loss * (active_loss.unsqueeze(-1))
                end_loss = torch.sum(end_loss)/torch.sum(active_loss)


            else:
                start_loss = loss_fct(start_logits.view(-1, self.num_labels), start_labels.view(-1))
                end_loss = loss_fct(end_logits.view(-1, self.num_labels), end_labels.view(-1))
            outputs = (start_loss+ end_loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

class BertForTokenBinaryClassificationWithTrigger(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.end_classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_labels=None, # batch* num_class* seq_length
        end_labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(token_type_ids),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # add triggrt embedding
        for i in range(sequence_output.size(0)):
            trigger_output=[]
            for j in range(sequence_output.size(1)):
                if token_type_ids[i][j]:
                    trigger_output.append(sequence_output[i][j])
            if trigger_output==[]: 
                print("segment_id == none")
                continue
            trigger_output = torch.stack(trigger_output,dim=0)
            trigger_output = torch.mean(trigger_output,dim=0)
            sequence_output[i] += trigger_output

        sequence_output = self.dropout(sequence_output)

        start_logits = self.start_classifier(sequence_output)
        end_logits = self.end_classifier(sequence_output)

        outputs = ([start_logits, end_logits],) + outputs[2:]  # add hidden states and attention if they are here
        if start_labels is not None and end_labels is not None:
            # loss_fct = CrossEntropyLoss()
            # loss_fct = FocalLoss(class_num=self.num_labels)
            loss_fct = BCEWithLogitsLoss(reduction="none")
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_start_logits = start_logits.view(-1, self.num_labels)
                active_end_logits = end_logits.view(-1, self.num_labels)

                active_start_labels = start_labels.view(-1, self.num_labels)
                active_end_labels = end_labels.view(-1, self.num_labels)
                # attention_mask: 
                # ignore_index: [cls], [sep]
                # non_index: no label

                # print(active_loss, active_loss.shape, \
                #      active_logits,active_logits.shape,\
                #      active_labels,active_labels.shape,\
                #      labels, labels.shape)
                #2048 2048*435 2048 8*256 
                start_loss = loss_fct(active_start_logits, active_start_labels.float())
                start_loss = start_loss * (active_loss.unsqueeze(-1))
                start_loss = torch.sum(start_loss)/torch.sum(active_loss)

                end_loss = loss_fct(active_end_logits, active_end_labels.float())
                end_loss = end_loss * (active_loss.unsqueeze(-1))
                end_loss = torch.sum(end_loss)/torch.sum(active_loss)


            else:
                start_loss = loss_fct(start_logits.view(-1, self.num_labels), start_labels.view(-1))
                end_loss = loss_fct(end_logits.view(-1, self.num_labels), end_labels.view(-1))
            outputs = (start_loss+ end_loss,) + () + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

