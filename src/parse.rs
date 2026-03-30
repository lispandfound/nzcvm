use winnow::ascii::{alpha1, alphanumeric1, escaped, float, multispace1, space1};
use winnow::combinator::{alt, delimited, fail, preceded, repeat, separated, separated_pair};
use winnow::error::{ContextError, StrContext, StrContextValue};
use winnow::prelude::*;
use winnow::stream::Accumulate;
use winnow::token::{none_of, one_of, take_till, take_while};
use winnow::Result;

// --- AST Definitions ---

#[derive(Debug, PartialEq)]
pub enum MaterialCommand {
    Block(BlockCmd),
    Grid(GridCmd), // Extensible for PFile, SFile, etc.
}

#[derive(Debug, PartialEq, Default)]
pub struct BlockCmd {
    pub vp: f64,
    pub vs: f64,
    pub rho: f64,
    pub vpgrad: Option<f64>,
    pub z1: Option<f64>,
    pub z2: Option<f64>,
    pub dx: Option<f64>,
    pub dz: Option<f64>,
}

impl Accumulate<BlockArg> for BlockCmd {
    fn initial(capacity: Option<usize>) -> Self {
        BlockCmd::default()
    }

    fn accumulate(&mut self, acc: BlockArg) {
        match acc {
            BlockArg::Vp(vp) => self.vp = vp,
            BlockArg::Vs(vs) => self.vs = vs,
            BlockArg::Rho(rho) => self.rho = rho,
            BlockArg::VpGrad(vpgrad) => self.vpgrad = Some(vpgrad),
            BlockArg::Z1(z1) => self.z1 = Some(z1),
            BlockArg::Z2(z2) => self.z2 = Some(z2),
            BlockArg::ResolutionHorizontal(dx) => self.dx = Some(dx),
            BlockArg::ResolutionVertical(dz) => self.dz = Some(dz),
        }
    }
}

// An intermediate enum to represent a parsed key-value pair for a block
#[derive(Debug, Clone, Copy)]
enum BlockArg {
    Vp(f64),
    Vs(f64),
    Rho(f64),
    VpGrad(f64),
    Z1(f64),
    Z2(f64),
    ResolutionVertical(f64),
    ResolutionHorizontal(f64),
}

#[derive(Debug, PartialEq, Default)]
pub struct GridCmd {
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub z: Option<f64>,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    pub mlat: Option<f64>,
    pub mlon: Option<f64>,
    pub az: Option<f64>,
    pub scale: Option<f64>,
    pub lat_p: Option<f64>,
    pub lon_p: Option<f64>,
    pub proj: Option<String>,
    pub ellps: Option<String>,
    pub datum: Option<String>,
}

impl Accumulate<GridArg> for GridCmd {
    fn initial(_capacity: Option<usize>) -> Self {
        GridCmd::default()
    }

    fn accumulate(&mut self, acc: GridArg) {
        match acc {
            GridArg::X(v) => self.x = Some(v),
            GridArg::Y(v) => self.y = Some(v),
            GridArg::Z(v) => self.z = Some(v),
            GridArg::Lat(v) => self.lat = Some(v),
            GridArg::Lon(v) => self.lon = Some(v),
            GridArg::Mlat(v) => self.mlat = Some(v),
            GridArg::Mlon(v) => self.mlon = Some(v),
            GridArg::Az(v) => self.az = Some(v),
            GridArg::Scale(v) => self.scale = Some(v),
            GridArg::LatP(v) => self.lat_p = Some(v),
            GridArg::LonP(v) => self.lon_p = Some(v),
            GridArg::Proj(v) => self.proj = Some(v),
            GridArg::Ellps(v) => self.ellps = Some(v),
            GridArg::Datum(v) => self.datum = Some(v),
        }
    }
}

#[derive(Debug, Clone)]
enum GridArg {
    X(f64),
    Y(f64),
    Z(f64),
    Lat(f64),
    Lon(f64),
    Mlat(f64),
    Mlon(f64),
    Az(f64),
    Scale(f64),
    LatP(f64),
    LonP(f64),
    Proj(String),
    Ellps(String),
    Datum(String),
}

fn parse_quoted(input: &mut &str) -> Result<String> {
    delimited(
        '"',
        escaped(
            none_of(['\"', '\\', '\n']).context(StrContext::Expected(
                StrContextValue::Description("any character except \", \\n or \\"),
            )),
            '\\',
            alt((
                "\\".value("\\"),
                "\"".value("\""),
                "n".value("\n"),
                "t".value("\t"),
            ))
            .context(StrContext::Expected(StrContextValue::Description(
                "valid escape character",
            ))),
        ),
        '"',
    )
    .context(StrContext::Expected(StrContextValue::Description(
        "string literal",
    )))
    .parse_next(input)
}

fn grid_arg(input: &mut &str) -> Result<GridArg> {
    let key = take_while(1.., |c: char| c.is_alphanumeric() || c == '_').parse_next(input)?;

    let _ = '='.parse_next(input)?;
    println!("key = {}", key);
    match key {
        "x" => float.map(GridArg::X).parse_next(input),
        "y" => float.map(GridArg::Y).parse_next(input),
        "z" => float.map(GridArg::Z).parse_next(input),
        "lat" => float.map(GridArg::Lat).parse_next(input),
        "lon" => float.map(GridArg::Lon).parse_next(input),
        "mlat" => float.map(GridArg::Mlat).parse_next(input),
        "mlon" => float.map(GridArg::Mlon).parse_next(input),
        "az" => float.map(GridArg::Az).parse_next(input),
        "scale" => float.map(GridArg::Scale).parse_next(input),
        "lat_p" => float.map(GridArg::LatP).parse_next(input),
        "lon_p" => float.map(GridArg::LonP).parse_next(input),
        "proj" => parse_quoted.map(GridArg::Proj).parse_next(input),
        "ellps" => parse_quoted.map(GridArg::Ellps).parse_next(input),
        "datum" => parse_quoted.map(GridArg::Datum).parse_next(input),
        _ => fail
            .context(StrContext::Label("Failed to parse grid keyword argument"))
            .parse_next(input),
    }
}

fn block_arg(input: &mut &str) -> Result<BlockArg> {
    let key = take_while(1.., |c: char| c.is_alphanumeric() || c == '_').parse_next(input)?;
    let _ = '='.parse_next(input)?;

    match key {
        "vp" => float.map(BlockArg::Vp).parse_next(input),
        "vs" => float.map(BlockArg::Vs).parse_next(input),
        "rho" => float.map(BlockArg::Rho).parse_next(input),
        "vpgrad" => float.map(BlockArg::VpGrad).parse_next(input),
        "z1" => float.map(BlockArg::Z1).parse_next(input),
        "z2" => float.map(BlockArg::Z2).parse_next(input),
        "dx" => float.map(BlockArg::ResolutionHorizontal).parse_next(input),
        "dz" => float.map(BlockArg::ResolutionVertical).parse_next(input),
        _ => fail
            .context(StrContext::Label("Failed to parse block keyword argument"))
            .parse_next(input),
    }
}

fn grid_command(input: &mut &str) -> Result<GridCmd> {
    let _ = "grid".parse_next(input)?;
    let _ = space1.parse_next(input)?;
    separated(1.., grid_arg, space1).parse_next(input)
}

fn block_command(input: &mut &str) -> Result<BlockCmd> {
    let _ = "block".parse_next(input)?;
    let _ = space1.parse_next(input)?;
    separated(1.., block_arg, space1).parse_next(input)
}
/// Parses a comment starting with `#` to the end of the line
fn comment<'a>(input: &mut &'a str) -> Result<&'a str> {
    preceded("#", take_till(0.., |c| c == '\n')).parse_next(input)
}

/// Consumes all whitespace and comments
fn ws_or_comment<'a>(input: &mut &'a str) -> Result<()> {
    let _ = repeat::<_, _, (), _, _>(0.., alt((multispace1, comment))).parse_next(input)?;
    Ok(())
}

fn keyword_f64<'a>(kw: &'static str) -> impl Parser<&'a str, f64, winnow::error::ContextError> {
    separated_pair(kw, "=", float).map(|(_, val)| val)
}

/// The main parser entry point
pub fn parse_material_model<'a>(input: &mut &'a str) -> Result<Vec<MaterialCommand>> {
    let mut commands = Vec::new();

    loop {
        let _ = ws_or_comment.parse_next(input)?;

        if input.is_empty() {
            break;
        }

        // Peek at the command name
        let cmd_name = take_while(1.., |c: char| c.is_alphabetic()).parse_peek(*input)?;

        match cmd_name.1 {
            "block" => {
                let cmd = block_command.parse_next(input)?;
                commands.push(MaterialCommand::Block(cmd));
            }
            "grid" => {
                let cmd = grid_command.parse_next(input)?;
                commands.push(MaterialCommand::Grid(cmd));
            }
            _ => {
                // Skip unknown or unimplemented commands to prevent infinite looping
                let _ = take_till(0.., |c| c == '\n').parse_next(input)?;
            }
        }
    }

    Ok(commands)
}

// --- Unit Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_f64() {
        let mut input = "vp=4000.5";
        let res = keyword_f64("vp").parse_next(&mut input).unwrap();
        assert_eq!(res, 4000.5);
    }

    #[test]
    fn test_block_arg() {
        let mut input = "rho=2000";
        let res = block_arg.parse_next(&mut input).unwrap();
        assert!(matches!(res, BlockArg::Rho(2000.0)));
    }

    #[test]
    fn test_quoted_str() {
        // Test parsing of quoted strings
        let mut input = "\"test\"";
        let res = parse_quoted(&mut input).unwrap();
        assert_eq!(res, "test");
    }

    #[test]
    fn test_quoted_escaped_str() {
        // Test parsing of quoted strings
        let mut input = "\"test\\twith quotes\"";
        println!("Input = {}", input);
        let res = parse_quoted(&mut input).unwrap();
        assert_eq!(res, "test\twith quotes");
    }

    #[test]
    fn test_block_command_unordered() {
        // Tests that the fold correctly handles arguments in any order
        let mut input = "block z1=15000 rho=2700 vs=3500 vp=6000 vpgrad=-0.01";
        let cmd = block_command.parse_next(&mut input).unwrap();

        assert_eq!(
            cmd,
            BlockCmd {
                vp: 6000.0,
                vs: 3500.0,
                rho: 2700.0,
                vpgrad: Some(-0.01),
                z1: Some(15000.0),
                z2: None,
            }
        );
    }

    #[test]
    fn test_grid_command_unordered() {
        // Tests the grid command handles arguments correctly
        let mut input = "grid x=10.0 y=20.0 z=30.0 proj=\"test projection\"";
        let cmd = grid_command.parse_next(&mut input).unwrap();
        let mut grid_default = GridCmd::default();
        grid_default.x = Some(10.0);
        grid_default.y = Some(20.0);
        grid_default.z = Some(30.0);
        grid_default.proj = Some("test projection".to_string());
        assert_eq!(cmd, grid_default);
    }

    #[test]
    fn test_parse_material_model_with_skips_and_comments() {
        let mut input = "
            # First layer
            block vp=4000 vs=2500 rho=2000
            
            
            # Second layer
            block vp=6000 vs=3500 rho=2700 z1=15000
        ";

        let cmds = parse_material_model.parse_next(&mut input).unwrap();
        assert_eq!(cmds.len(), 2);

        assert_eq!(
            cmds[0],
            MaterialCommand::Block(BlockCmd {
                vp: 4000.0,
                vs: 2500.0,
                rho: 2000.0,
                ..Default::default()
            })
        );

        assert_eq!(
            cmds[1],
            MaterialCommand::Block(BlockCmd {
                vp: 6000.0,
                vs: 3500.0,
                rho: 2700.0,
                z1: Some(15000.0),
                ..Default::default()
            })
        );
    }
}
