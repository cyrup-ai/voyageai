use serde::{Deserialize, Serialize};
use syn::{File, Item};
use pulldown_cmark::{Parser, Event, Tag};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBlock {
    pub language: Option<String>,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstNode {
    pub kind: String,
    pub content: String,
    pub children: Vec<AstNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeEmbedding {
    pub text_embedding: Vec<f32>,
    pub ast_embedding: Vec<f32>,
}

pub fn extract_code_blocks(markdown: &str) -> Vec<CodeBlock> {
    let parser = Parser::new(markdown);
    let mut blocks = Vec::new();
    let mut current_block = None;

    for event in parser {
        match event {
            Event::Start(Tag::CodeBlock(lang)) => {
                current_block = Some(CodeBlock {
                    language: lang.map(|s| s.to_string()),
                    content: String::new(),
                });
            }
            Event::Text(text) => {
                if let Some(block) = &mut current_block {
                    block.content.push_str(&text);
                }
            }
            Event::End(Tag::CodeBlock(_)) => {
                if let Some(block) = current_block.take() {
                    blocks.push(block);
                }
            }
            _ => {}
        }
    }
    blocks
}

pub fn parse_rust_ast(code: &str) -> Result<Vec<AstNode>, syn::Error> {
    let syntax = syn::parse_file(code)?;
    Ok(parse_items(&syntax.items))
}

fn parse_items(items: &[Item]) -> Vec<AstNode> {
    items.iter().map(|item| parse_item(item)).collect()
}

fn parse_item(item: &Item) -> AstNode {
    match item {
        Item::Fn(f) => AstNode {
            kind: "function".to_string(),
            content: f.sig.ident.to_string(),
            children: Vec::new(),
        },
        Item::Struct(s) => AstNode {
            kind: "struct".to_string(), 
            content: s.ident.to_string(),
            children: Vec::new(),
        },
        Item::Enum(e) => AstNode {
            kind: "enum".to_string(),
            content: e.ident.to_string(),
            children: Vec::new(),
        },
        Item::Impl(i) => AstNode {
            kind: "impl".to_string(),
            content: format!("{:?}", i.self_ty),
            children: parse_items(&i.items),
        },
        _ => AstNode {
            kind: "other".to_string(),
            content: String::new(),
            children: Vec::new(),
        }
    }
}
