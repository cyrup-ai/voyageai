use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SerializableAst {
    pub items: Vec<Item>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Item {
    Function(Function),
    Struct(Struct),
    Enum(Enum),
    Module(Module),
    Use(Use),
    Other(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub visibility: Option<String>,
    pub inputs: Vec<String>,
    pub output: Option<String>,
    pub is_async: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Struct {
    pub name: String,
    pub visibility: Option<String>,
    pub fields: Vec<Field>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub ty: String,
    pub visibility: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Enum {
    pub name: String,
    pub visibility: Option<String>,
    pub variants: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Module {
    pub name: String,
    pub visibility: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Use {
    pub path: String,
}