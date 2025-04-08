use crate::models::ast::*;
use quote::ToTokens;
use syn::{Item as SynItem, ItemEnum, ItemFn, ItemMod, ItemStruct, ItemUse};

pub fn parse_rust_ast(code: &str) -> Result<SerializableAst, syn::Error> {
    let file = syn::parse_file(code)?;
    let items = file.items.into_iter().map(convert_item).collect();
    Ok(SerializableAst { items })
}

fn convert_item(item: SynItem) -> Item {
    match item {
        SynItem::Fn(f) => Item::Function(convert_function(f)),
        SynItem::Struct(s) => Item::Struct(convert_struct(s)),
        SynItem::Enum(e) => Item::Enum(convert_enum(e)),
        SynItem::Mod(m) => Item::Module(convert_module(m)),
        SynItem::Use(u) => Item::Use(convert_use(u)),
        other => Item::Other(other.to_token_stream().to_string()),
    }
}

fn convert_function(f: ItemFn) -> Function {
    Function {
        name: f.sig.ident.to_string(),
        visibility: Some(f.vis.to_token_stream().to_string()),
        inputs: f
            .sig
            .inputs
            .iter()
            .map(|arg| arg.to_token_stream().to_string())
            .collect(),
        output: match &f.sig.output {
            syn::ReturnType::Default => None,
            syn::ReturnType::Type(_, ty) => Some(ty.to_token_stream().to_string()),
        },
        is_async: f.sig.asyncness.is_some(),
    }
}

fn convert_struct(s: ItemStruct) -> Struct {
    Struct {
        name: s.ident.to_string(),
        visibility: Some(s.vis.to_token_stream().to_string()),
        fields: s
            .fields
            .iter()
            .map(|f| Field {
                name: f
                    .ident
                    .as_ref()
                    .map(|i| i.to_string())
                    .unwrap_or_default(),
                ty: f.ty.to_token_stream().to_string(),
                visibility: Some(f.vis.to_token_stream().to_string()),
            })
            .collect(),
    }
}

fn convert_enum(e: ItemEnum) -> Enum {
    Enum {
        name: e.ident.to_string(),
        visibility: Some(e.vis.to_token_stream().to_string()),
        variants: e.variants.iter().map(|v| v.ident.to_string()).collect(),
    }
}

fn convert_module(m: ItemMod) -> Module {
    Module {
        name: m.ident.to_string(),
        visibility: Some(m.vis.to_token_stream().to_string()),
    }
}

fn convert_use(u: ItemUse) -> Use {
    Use {
        path: u.tree.to_token_stream().to_string(),
    }
}

pub struct CodeBlock {
    pub language: Option<String>,
    pub content: String,
}

impl CodeBlock {
    pub fn new(language: Option<String>, content: String) -> Self {
        Self { language, content }
    }

    pub fn parse(&self) -> Result<SerializableAst, syn::Error> {
        match self.language.as_deref() {
            Some("rust") | Some("rs") => parse_rust_ast(&self.content),
            _ => Ok(SerializableAst { items: vec![] }),
        }
    }
}

pub fn extract_code_blocks(markdown: &str) -> Vec<CodeBlock> {
    let mut blocks = Vec::new();
    let mut lines = markdown.lines().peekable();

    while let Some(line) = lines.next() {
        if line.starts_with("```") {
            let language = line.trim_start_matches('`').trim().to_string();
            let language = if language.is_empty() {
                None
            } else {
                Some(language)
            };

            let mut content = String::new();
            for content_line in lines.by_ref() {
                if content_line.starts_with("```") {
                    break;
                }
                content.push_str(content_line);
                content.push('\n');
            }
            blocks.push(CodeBlock::new(language, content));
        }
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rust_ast() {
        let code = r#"
            struct Test {
                field: String,
            }

            fn example() -> String {
                "test".to_string()
            }
        "#;

        let ast = parse_rust_ast(code).unwrap();
        assert_eq!(ast.items.len(), 2);
    }

    #[test]
    fn test_extract_code_blocks() {
        let markdown = r#"
Some text

```rust
fn test() {}
```

```
plain text
```
"#;

        let blocks = extract_code_blocks(markdown);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].language, Some("rust".to_string()));
        assert!(blocks[0].content.contains("fn test()"));
    }
}
