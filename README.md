# KRX_competition 2024
본 레포지터리는 [제3회 KRX 금융 언어 모델 경진대회](https://krxbench.koscom.co.kr/home/main) 참가 후기 및 소스코드의 정리를 위한 저장소입니다.

## Used Base model

- [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B)
- [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b)
- [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)

## Fine-Tuning model

- [Model list](https://huggingface.co/vitus48683)

## 참조 데이터셋

- [금융 분야 다국어 병렬 말뭉치 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71782)
- [amphora/rewrite-se-quant](https://huggingface.co/datasets/amphora/rewrite-se-quant)
- [amphora/krx-sample-instructions](https://huggingface.co/datasets/amphora/krx-sample-instructions)
- [숫자연산 기계독해 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71568)

## 학습환경
- 24 vCPU, 128GB Ram, A100 PCIe 80GB 서버 임대

## 참고자료

- [KRX 튜토리얼](https://apricot-behavior-a37.notion.site/107a57df933c80aaafb7c51f8dcec06c)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [Mergekit](https://github.com/arcee-ai/mergekit)
