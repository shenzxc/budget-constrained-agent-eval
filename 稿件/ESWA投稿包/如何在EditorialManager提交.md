# ESWA 提交步骤(Editorial Manager)

投稿系统:https://www.editorialmanager.com/eswa/
本文件夹已备好全部要上传的文件。按顺序走即可。

## 一、注册/登录
- 用邮箱注册 Author 账号(建议用 19708110566@139.com,与稿件一致)。
- 登录后进 "Submit New Manuscript"。

## 二、逐屏填写(EM会一屏一屏问)

1. **Article Type**:选 `Research Paper`(或 Full Length Article)。
2. **Title**:粘贴
   `Budget-Constrained Evaluation of Open-Weight LLM Agents: Success–Budget Curves, Price Reversal, and Thinking-Budget Saturation`
3. **Abstract**:从 manuscript.pdf 首页复制摘要全文粘贴。
4. **Keywords**:逐个输入(共7个)
   LLM agents;budget-constrained evaluation;cost-aware decision support;intelligent systems deployment;inference cost;open-weight models;tool-use agents
5. **Authors**:只有你一人=通讯作者
   - Given name: Weiming;Family name: Shen
   - Email: 19708110566@139.com
   - Affiliation: Independent Researcher, Suqian, Jiangsu, China
   - ORCID: 0009-0006-8222-1668(系统可能让你关联ORCID账号,点授权即可)
6. **Suggested reviewers**(如系统要求):可留空或填2-3个该领域公开学者(非必填时跳过)。
7. **Declarations / Questions**:
   - Conflict of interest:None(稿件内已有 Declarations 段)
   - Funding:None
   - Data availability:已公开,Zenodo DOI 10.5281/zenodo.21215799
   - Generative AI use:已在稿件 Declarations 中声明

## 三、上传文件(Attach Files)——用本文件夹里的

| EM 文件类型(Item) | 上传哪个文件 |
|---|---|
| Manuscript | `manuscript.pdf` |
| Highlights | `Highlights.txt`(文件名含 highlights,ESWA必需) |
| Cover Letter | `CoverLetter.txt`(或粘贴进 Cover Letter 文本框) |

> 说明:本稿把 Declarations、CRediT、Data Availability 都写进了正文 PDF,所以不需要单独的声明文件;若系统强制要求单独的 "Declaration of Interest" 文件,把 CoverLetter 里相应句子另存一个 txt 传上去即可,或直接在系统文本框写 "The author declares no competing interests."

## 四、提交前检查(EM会生成一个PDF让你Approve)
- 确认 PDF 首页作者/单位/ORCID 正确;
- 确认 Highlights 显示为5条;
- 点 "Approve Submission" → 完成。

## 五、提交后
- 系统会给你一个 Manuscript Number(形如 ESWA-D-26-xxxxx),记下来;
- 状态会从 "Submitted to Journal" → "With Editor" → "Under Review"。
- 有任何一步卡住,截图问我。

## 备注:若期刊要求 LaTeX 源码
ESWA 接受 PDF 初审;若进入修改/接收阶段要源码,打包 `稿件/latex/` 整个文件夹(manuscript.tex + refs.bib + elsarticle*.cls/.bst + 图)即可。图在 experiments/analysis/output/figures/,打包时把用到的 fig1–5 的 pdf 一并放进 latex 目录并把 \graphicspath 改成当前目录。
