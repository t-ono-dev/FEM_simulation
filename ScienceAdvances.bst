%%
%% This is file `ScienceAdvances_custom.bst',
%% generated from ScienceAdvances.bst and customized per user requirements:
%%  - Use journal abbreviations for articles (e.g., Phys. Rev. Lett.)
%%  - Omit publisher information
%%  - Omit page counts for books (book entries already do not print pages)
%%  - Keep chapters/pages for inbook/incollection
%%
%% Copyright base: 1994-1999 Patrick W Daly (merlin.mbs / physjour.mbs)
%% Modified 2014 for Science Advances by Aaron Clauset
%% Further customized 2025-11-22
%%
%% This bibliography style file is intended for texts in ENGLISH.
%% Numerical citation style.
%%
ENTRY
  { address
    author
    booktitle
    chapter
    edition
    editor
    howpublished
    institution
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    volume
    year
    url
  }
  {}
  { label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}

STRINGS { s t u }


FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
        { add.period$ write$
          newline$
          "\newblock " write$
        }
        { output.state before.all =
            'write$
            { add.period$ " " * write$ }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}

FUNCTION {add.blank}
{  " " * before.all 'output.state :=
}

FUNCTION {date.block}
{
  skip$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}

FUNCTION {new.block.checka}
{ empty$
    'skip$
    'new.block
  if$
}

FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}

FUNCTION {new.sentence.checka}
{ empty$
    'skip$
    'new.sentence
  if$
}

FUNCTION {new.sentence.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.sentence
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "{\it " swap$ * "\/}" * }
  if$
}

FUNCTION {bolden}
{ duplicate$ empty$
    { pop$ "" }
    { "{\bf " swap$ * "}" * }
  if$
}

FUNCTION {capitalize}
{ "u" change.case$ "t" change.case$ }

FUNCTION {space.word}
{ " " swap$ * " " * }

FUNCTION {bbl.and} { "and" }
FUNCTION {bbl.etal} { "et~al." }
FUNCTION {bbl.editors} { "eds." }
FUNCTION {bbl.editor} { "ed." }
FUNCTION {bbl.edby} { "edited by" }
FUNCTION {bbl.edition} { "edition." }
FUNCTION {bbl.volume} { "vol." }
FUNCTION {bbl.of} { "of" }
FUNCTION {bbl.number} { "no." }
FUNCTION {bbl.nr} { "no." }
FUNCTION {bbl.in} { "in" }
FUNCTION {bbl.pages} { "pp." }
FUNCTION {bbl.page} { "p." }
FUNCTION {bbl.chapter} { "chap." }
FUNCTION {bbl.techrep} { "Tech. Rep." }
FUNCTION {bbl.mthesis} { "Master's thesis" }
FUNCTION {bbl.phdthesis} { "Ph.D. thesis" }
FUNCTION {bbl.first} { "First" }
FUNCTION {bbl.second} { "Second" }
FUNCTION {bbl.third} { "Third" }
FUNCTION {bbl.fourth} { "Fourth" }
FUNCTION {bbl.fifth} { "Fifth" }
FUNCTION {bbl.st} { "st" }
FUNCTION {bbl.nd} { "nd" }
FUNCTION {bbl.rd} { "rd" }
FUNCTION {bbl.th} { "th" }

MACRO {jan} {"Jan."}
MACRO {feb} {"Feb."}
MACRO {mar} {"Mar."}
MACRO {apr} {"Apr."}
MACRO {may} {"May"}
MACRO {jun} {"Jun."}
MACRO {jul} {"Jul."}
MACRO {aug} {"Aug."}
MACRO {sep} {"Sep."}
MACRO {oct} {"Oct."}
MACRO {nov} {"Nov."}
MACRO {dec} {"Dec."}

FUNCTION {eng.ord}
{ duplicate$ "1" swap$ *
  #-2 #1 substring$ "1" =
     { bbl.th * }
     { duplicate$ #-1 #1 substring$
       duplicate$ "1" =
         { pop$ bbl.st * }
         { duplicate$ "2" =
             { pop$ bbl.nd * }
             { "3" =
                 { bbl.rd * }
                 { bbl.th * }
               if$
             }
           if$
          }
       if$
     }
   if$
}

MACRO {aa}{"Astron. \& Astrophys."}
MACRO {aasup}{"Astron. \& Astrophys. Suppl. Ser."}
MACRO {aph} {"Acta Phys."}
MACRO {advp} {"Adv. Phys."}
MACRO {ajp} {"Amer. J. Phys."}
MACRO {ajm} {"Amer. J. Math."}
MACRO {amsci} {"Amer. Sci."}
MACRO {anofd} {"Ann. Fluid Dyn."}
MACRO {am} {"Ann. Math."}
MACRO {ap} {"Ann. Phys. (NY)"}
MACRO {adp} {"Ann. Phys. (Leipzig)"}
MACRO {ao} {"Appl. Opt."}
MACRO {apl} {"Appl. Phys. Lett."}
MACRO {app} {"Astroparticle Phys."}
MACRO {apj} {"Astrophys. J."}
MACRO {apjsup} {"Astrophys. J. Suppl. Ser."}
MACRO {baps} {"Bull. Amer. Phys. Soc."}
MACRO {cmp} {"Comm. Math. Phys."}
MACRO {cpam} {"Commun. Pure Appl. Math."}
MACRO {cppcf} {"Comm. Plasma Phys. \& Controlled Fusion"}
MACRO {cpc} {"Comp. Phys. Comm."}
MACRO {cqg} {"Class. Quant. Grav."}
MACRO {cra} {"C. R. Acad. Sci. A"}
MACRO {fed} {"Fusion Eng. \& Design"}
MACRO {ft} {"Fusion Tech."}
MACRO {grg} {"Gen. Relativ. Gravit."}
MACRO {ieeens} {"IEEE Trans. Nucl. Sci."}
MACRO {ieeeps} {"IEEE Trans. Plasma Sci."}
MACRO {ijimw} {"Interntl. J. Infrared \& Millimeter Waves"}
MACRO {ip} {"Infrared Phys."}
MACRO {irp} {"Infrared Phys."}
MACRO {jap} {"J. Appl. Phys."}
MACRO {jasa} {"J. Acoust. Soc. America"}
MACRO {jcp} {"J. Comp. Phys."}
MACRO {jetp} {"Sov. Phys.--JETP"}
MACRO {jfe} {"J. Fusion Energy"}
MACRO {jfm} {"J. Fluid Mech."}
MACRO {jmp} {"J. Math. Phys."}
MACRO {jne} {"J. Nucl. Energy"}
MACRO {jnec} {"J. Nucl. Energy, C: Plasma Phys., Accelerators, Thermonucl. Res."}
MACRO {jnm} {"J. Nucl. Mat."}
MACRO {jpc} {"J. Phys. Chem."}
MACRO {jpp} {"J. Plasma Phys."}
MACRO {jpsj} {"J. Phys. Soc. Japan"}
MACRO {jsi} {"J. Sci. Instrum."}
MACRO {jvst} {"J. Vac. Sci. \& Tech."}
MACRO {nat} {"Nature"}
MACRO {nature} {"Nature"}
MACRO {nedf} {"Nucl. Eng. \& Design/Fusion"}
MACRO {nf} {"Nucl. Fusion"}
MACRO {nim} {"Nucl. Inst. \& Meth."}
MACRO {nimpr} {"Nucl. Inst. \& Meth. in Phys. Res."}
MACRO {np} {"Nucl. Phys."}
MACRO {npb} {"Nucl. Phys. B"}
MACRO {nt/f} {"Nucl. Tech./Fusion"}
MACRO {npbpc} {"Nucl. Phys. B (Proc. Suppl.)"}
MACRO {inc} {"Nuovo Cimento"}
MACRO {nc} {"Nuovo Cimento"}
MACRO {pf} {"Phys. Fluids"}
MACRO {pfa} {"Phys. Fluids A: Fluid Dyn."}
MACRO {pfb} {"Phys. Fluids B: Plasma Phys."}
MACRO {pl} {"Phys. Lett."}
MACRO {pla} {"Phys. Lett. A"}
MACRO {plb} {"Phys. Lett. B"}
MACRO {prep} {"Phys. Rep."}
MACRO {pnas} {"Proc. Nat. Acad. Sci. USA"}
MACRO {pp} {"Phys. Plasmas"}
MACRO {ppcf} {"Plasma Phys. \& Controlled Fusion"}
MACRO {phitrsl} {"Philos. Trans. Roy. Soc. London"}
MACRO {prl} {"Phys. Rev. Lett."}
MACRO {pr} {"Phys. Rev."}
MACRO {physrev} {"Phys. Rev."}
MACRO {pra} {"Phys. Rev. A"}
MACRO {prb} {"Phys. Rev. B"}
MACRO {prc} {"Phys. Rev. C"}
MACRO {prd} {"Phys. Rev. D"}
MACRO {pre} {"Phys. Rev. E"}
MACRO {ps} {"Phys. Scripta"}
MACRO {procrsl} {"Proc. Roy. Soc. London"}
MACRO {rmp} {"Rev. Mod. Phys."}
MACRO {rsi} {"Rev. Sci. Inst."}
MACRO {science} {"Science"}
MACRO {sciam} {"Sci. Am."}
MACRO {sam} {"Stud. Appl. Math."}
MACRO {sjpp} {"Sov. J. Plasma Phys."}
MACRO {spd} {"Sov. Phys.--Doklady"}
MACRO {sptp} {"Sov. Phys.--Tech. Phys."}
MACRO {spu} {"Sov. Phys.--Uspeki"}
MACRO {acmcs} {"ACM Comput. Surv."}
MACRO {acta} {"Acta Inf."}
MACRO {cacm} {"Commun. ACM"}
MACRO {ibmjrd} {"IBM J. Res. Dev."}
MACRO {ibmsj} {"IBM Syst.~J."}
MACRO {ieeese} {"IEEE Trans. Softw. Eng."}
MACRO {ieeetc} {"IEEE Trans. Comput."}
MACRO {ieeetcad} {"IEEE Trans. Comput.-Aided Design Integrated Circuits"}
MACRO {ipl} {"Inf. Process. Lett."}
MACRO {jacm} {"J.~ACM"}
MACRO {jcss} {"J.~Comput. Syst. Sci."}
MACRO {scp} {"Sci. Comput. Programming"}
MACRO {sicomp} {"SIAM J. Comput."}
MACRO {tocs} {"ACM Trans. Comput. Syst."}
MACRO {tods} {"ACM Trans. Database Syst."}
MACRO {tog} {"ACM Trans. Gr."}
MACRO {toms} {"ACM Trans. Math. Softw."}
MACRO {toois} {"ACM Trans. Office Inf. Syst."}
MACRO {toplas} {"ACM Trans. Prog. Lang. Syst."}
MACRO {tcs} {"Theoretical Comput. Sci."}

INTEGERS { nameptr namesleft numnames numnames.orig }


FUNCTION {format.names}
{ 's :=
  "" 't :=

  % 元の著者数を保存し、表示は最大20人までに制限
  s num.names$ 'numnames.orig :=
  numnames.orig 'numnames :=
  numnames #20 >
    { #20 'numnames := }
    'skip$
  if$

  #1 'nameptr :=

  % 1..numnames を出力
  { nameptr numnames > not }
  { s nameptr "{f.~}{vv~}{ll}{, jj}" format.name$ 'u :=

    nameptr #1 =
      { u 't := }
      {
        % ===== 20人超(=et al. を使う)なら "and" を一切入れない =====
        numnames.orig #20 >
          { t ", " * u * 't := }     % 常にカンマ区切り
          {
            % ===== 20人以下なら最後だけ and を挿入 =====
            nameptr numnames =
              { numnames #2 =
                  { t " and~" * u * 't := }    % 2人だけ：カンマ無し
                  { t ", and~" * u * 't := }   % 3人以上：通常
                if$
              }
              { t ", " * u * 't := }
            if$
          }
        if$
      }
    if$

    nameptr #1 + 'nameptr :=
  }
  while$

  % 20人超なら末尾に et al.（直前にカンマも and も付けない）
  numnames.orig #20 >
    { t " " * bbl.etal emphasize * 't := }  % ←「 ... Hirsh et~al.」
    'skip$
  if$

  t
}

  while$
}





FUNCTION {format.names.ed}
{ format.names }

FUNCTION {format.authors}
{ author empty$
    { "" }
    { author format.names }
  if$
}

FUNCTION {format.editors}
{ editor empty$
    { "" }
    { editor format.names
      ", " *
      editor num.names$ #1 >
        'bbl.editors
        'bbl.editor
      if$
      *
    }
  if$
}

FUNCTION {format.in.editors}
{ editor empty$
    { "" }
    { editor format.names.ed
    }
  if$
}

FUNCTION {format.note}
{
 note empty$
    { "" }
    { note #1 #1 substring$
      duplicate$ "{" =
        'skip$
        { output.state mid.sentence =
          { "l" }
          { "u" }
        if$
        change.case$
        }
      if$
      note #2 global.max$ substring$ *
    }
  if$
}

FUNCTION {quote.wrap}
{ duplicate$ empty$
    { pop$ "" }
    { "``" swap$ * "''" * }   % 左右のTeX引用符
  if$
}


FUNCTION {format.title}
{ title empty$
    { "" }
    { title "t" change.case$ quote.wrap }
  if$
}

FUNCTION {format.title.misc}
{ title empty$
    { "" }
    { title "t" change.case$ }    % no quotes for misc
  if$
}

FUNCTION {is.arxiv}
{ publisher empty$
    { howpublished empty$
        { note empty$
            { #0 }
            { note "l" change.case$ #1 #5 substring$ "arxiv" = }
          if$
        }
        { howpublished "l" change.case$ #1 #5 substring$ "arxiv" = }
      if$
    }
    { publisher "l" change.case$ #1 #5 substring$ "arxiv" = }
  if$
}

FUNCTION {format.arxiv.id}
{ publisher empty$
    { howpublished empty$
        { note empty$
            { "" }
            { note }
          if$
        }
        { howpublished }
      if$
    }
    { publisher }
  if$
}



FUNCTION {format.url}
{ url empty$
    { "" }
    { url }                       % print as-is
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {n.dashify}
{
  't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { "-" *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}

FUNCTION {word.in}
{ "" }

FUNCTION {format.date}
{ year empty$
    { "" }
      'year
  if$
  duplicate$ empty$
    'skip$
    {
      before.all 'output.state :=
    " (" swap$ * ")" *
    }
  if$
}

FUNCTION {format.btitle}
{ title empty$
    { booktitle empty$
        { "" }
        { booktitle "t" change.case$ emphasize }  % フォールバック
      if$
    }
    { title "t" change.case$ emphasize }          % @book は通常これ
  if$
}



FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}

FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}

FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { bbl.volume volume tie.or.space.connect
      series empty$
        'skip$
        { bbl.of space.word * series emphasize * }
      if$
      "volume and number" number either.or.check
    }
  if$
}

FUNCTION {format.number.series}
{ volume empty$
    { number empty$
        { series field.or.null }
        { output.state mid.sentence =
            { bbl.number }
            { bbl.number capitalize }
          if$
          number tie.or.space.connect
          series empty$
            { "there's a number but no series in " cite$ * warning$ }
            { bbl.in space.word * series * }
          if$
        }
      if$
    }
    { "" }
  if$
}

FUNCTION {is.num}
{ chr.to.int$
  duplicate$ "0" chr.to.int$ < not
  swap$ "9" chr.to.int$ > not and
}

FUNCTION {extract.num}
{ duplicate$ 't :=
  "" 's :=
  { t empty$ not }
  { t #1 #1 substring$
    t #2 global.max$ substring$ 't :=
    duplicate$ is.num
      { s swap$ * 's := }
      { pop$ "" 't := }
    if$
  }
  while$
  s empty$
    'skip$
    { pop$ s }
  if$
}

FUNCTION {convert.edition}
{ edition extract.num "l" change.case$ 's :=
  s "first" = s "1" = or
    { bbl.first 't := }
    { s "second" = s "2" = or
        { bbl.second 't := }
        { s "third" = s "3" = or
            { bbl.third 't := }
            { s "fourth" = s "4" = or
                { bbl.fourth 't := }
                { s "fifth" = s "5" = or
                    { bbl.fifth 't := }
                    { s #1 #1 substring$ is.num
                        { s eng.ord 't := }
                        { edition 't := }
                      if$
                    }
                  if$
                }
              if$
            }
          if$
        }
      if$
    }
  if$
  t
}
FUNCTION {num.to.ordinal}
{ 's :=
  "00" s * #-2 #2 substring$ 't :=
  t "11" =
  t "12" = or
  t "13" = or
    { s "th" * }
    { "0" s * #-1 #1 substring$ 't :=
      t "1" =
        { s "st" * }
        { t "2" =
            { s "nd" * }
            { t "3" =
                { s "rd" * }
                { s "th" * }
              if$
            }
          if$
        }
      if$
    }
  if$
}

FUNCTION {edition.to.num}
{ edition extract.num duplicate$ empty$
    { pop$
      edition "l" change.case$ 't :=
      t #1 #5 substring$ "first"  = { "1" }
      { t #1 #6 substring$ "second" = { "2" }
        { t #1 #5 substring$ "third"  = { "3" }
          { t #1 #6 substring$ "fourth" = { "4" }
            { t #1 #5 substring$ "fifth"  = { "5" }
              { "" }
              if$
            } if$
          } if$
        } if$
      } if$
    }
    'skip$
  if$
}

FUNCTION {format.edition.short}
{ edition empty$
    { "" }
    { edition.to.num duplicate$ empty$
        { pop$ "" }
        { num.to.ordinal " ed. " * }   % 2 -> 2nd ed.
      if$
    }
  if$
}

FUNCTION {format.book.paren}
{ publisher empty$ address empty$ and year empty$ and
    { "" }
    {
      "("
      publisher empty$ 'skip$ { publisher * } if$
      address empty$
        'skip$
        { publisher empty$ 'skip$ { ", " * } if$
          address *
        }
      if$
      year empty$
        'skip$
        { publisher empty$ address empty$ and 'skip$ { ", " * } if$
          year *
        }
      if$
      ")" *
    }
  if$
}



FUNCTION {format.book.tail}
{ publisher empty$ year empty$ and edition empty$ and
    { "" }
    {
      " ("
      publisher empty$ 'skip$ { publisher * } if$
      year empty$
        'skip$
        { publisher empty$ 'skip$ { ", " * } if$
          year *
        }
      if$
      edition empty$
        'skip$
        { publisher empty$ year empty$ and 'skip$ { ", " * } if$
          format.edition.short *
        }
      if$
      ")"*
    }
  if$
}



FUNCTION {format.edition}
{ edition empty$
    { "" }
    { output.state mid.sentence =
        { convert.edition "l" change.case$ " " * bbl.edition * }
        { convert.edition "t" change.case$ " " * bbl.edition * }
      if$
    }
  if$
}

INTEGERS { multiresult }

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
        { #1 'multiresult := }
        { t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
  multiresult
}

FUNCTION {format.pages}
{ pages empty$
    { "" }
    { pages first.page 't :=
      pages multi.page.check
        { bbl.pages t tie.or.space.connect } % 336--338 -> pp. 336
        { bbl.page  t tie.or.space.connect } % 336 -> p. 336
      if$
    }
  if$
}


FUNCTION {first.page}
{ 't :=
  ""
    {  t empty$ not t #1 #1 substring$ "-" = not and }
    { t #1 #1 substring$ *
      t #2 global.max$ substring$ 't :=
    }
  while$
}

FUNCTION {format.journal.pages}
{ pages empty$
    'skip$
    { duplicate$ empty$
        { pop$ format.pages }
        {
          ", " *
          pages first.page *
        }
      if$
    }
  if$
}


FUNCTION {format.vol.num.pages}
{ volume field.or.null
  bolden
  format.journal.pages
}

FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
        { bbl.chapter }
        { type "l" change.case$ }
      if$
      chapter tie.or.space.connect
      pages empty$
        'skip$
        { ", " * format.pages * }
      if$
    }
  if$
}

FUNCTION {format.in.ed.proctitle}
{ booktitle empty$
    { "" }
    { editor empty$
        { booktitle }                % ← emphasize しない
        { booktitle
          ", " *
          format.in.editors *
          ", " *
          editor num.names$ #1 >
            { bbl.editors }
            { bbl.editor }
          if$
          *
        }
      if$
    }
  if$
}




FUNCTION {format.in.ed.booktitle}
{ booktitle empty$
    { "" }
    { editor empty$
        { booktitle emphasize}
        { booktitle emphasize
          ", " *
          format.in.editors *
          ", " *
          editor num.names$ #1 >
            { bbl.editors }
            { bbl.editor }
          if$
          *
        }
      if$
    }
  if$
}

FUNCTION {empty.misc.check}
{ author empty$ title empty$ howpublished empty$
  month empty$ year empty$ note empty$
  and and and and and
    { "all relevant fields are empty in " cite$ * warning$ }
    'skip$
  if$
}

FUNCTION {format.thesis.type}
{ type empty$
    'skip$
    { pop$
      type "t" change.case$
    }
  if$
}

FUNCTION {format.tr.number}
{ type empty$
    { bbl.techrep }
    'type
  if$
  number empty$
    { "t" change.case$ }
    { number tie.or.space.connect }
  if$
}

FUNCTION {format.article.crossref}
{
  key empty$
    { journal empty$
        { "need key or journal for " cite$ * " to crossref " * crossref *
          warning$
          ""
        }
        { word.in journal emphasize * }
      if$
    }
    { word.in key * " " *}
  if$
  " \cite{" * crossref * "}" *
}

FUNCTION {format.crossref.editor}
{ editor #1 "{vv~}{ll}" format.name$
  editor num.names$ duplicate$
  #2 >
    { pop$
      " " * bbl.etal emphasize *
    }
    { #2 <
        'skip$
        { editor #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
            {
              " " * bbl.etal emphasize *
            }
            { bbl.and space.word * editor #2 "{vv~}{ll}" format.name$
              * }
          if$
        }
      if$
    }
  if$
}

FUNCTION {format.book.crossref}
{ volume empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      word.in
    }
    { bbl.volume volume tie.or.space.connect
      bbl.of space.word *
    }
  if$
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
        { series empty$
            { "need editor, key, or series for " cite$ * " to crossref " *
              crossref * warning$
              "" *
            }
            { series emphasize * }
          if$
        }
        { key * }
      if$
    }
    { format.crossref.editor * }
  if$
  " \cite{" * crossref * "}" *
}

FUNCTION {format.incoll.inproc.crossref}
{
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
        { booktitle empty$
            { "need editor, key, or booktitle for " cite$ * " to crossref " *
              crossref * warning$
              ""
            }
            { word.in booktitle emphasize * }
          if$
        }
        { word.in key * " " *}
      if$
    }
    { word.in format.crossref.editor * " " *}
  if$
  " \cite{" * crossref * "}" *
}

FUNCTION {format.org.or.pub}
{ 't :=
  ""
  year empty$
    { "empty year in " cite$ * warning$ }
    'skip$
  if$
  address empty$ t empty$ and
  year empty$ and
    'skip$
    {
      add.blank "(" *
      t empty$
        { address empty$
          'skip$
          { address * }
          if$
        }
        { t *
          address empty$
            'skip$
            { ", " * address * }
          if$
        }
      if$
      year empty$
        'skip$
        { t empty$ address empty$ and
            'skip$
            { ", " * }
          if$
          year *
        }
      if$
      ")" *
    }
  if$
}

% ======= USER REQUEST: omit publisher info =======
FUNCTION {format.publisher.address}
{ "" }

FUNCTION {format.organization.address}
{ organization empty$
    { "" }
    { organization }
  if$
  format.org.or.pub
}

FUNCTION {new.comma}
{ output.state before.all =
    'skip$
    { mid.sentence 'output.state := }
  if$
}

% ======= USER REQUEST: abbreviate journal names =======
FUNCTION {journal.to.abbrev}
{ journal duplicate$ empty$
    { "" }
    {
      "l" change.case$ 's :=
      s "physical review letters" =
        { "Phys. Rev. Lett." }
        { s "physical review a" =
            { "Phys. Rev. A" }
            { s "physical review b" =
                { "Phys. Rev. B" }
                { s "physical review c" =
                    { "Phys. Rev. C" }
                    { s "physical review d" =
                        { "Phys. Rev. D" }
                        { s "physical review e" =
                            { "Phys. Rev. E" }
                            { s "applied physics letters" =
                                { "Appl. Phys. Lett." }
                                { s "journal of applied physics" =
                                    { "J. Appl. Phys." }
                                    { s "nature nanotechnology" =
                                        { "Nat. Nanotechnol." }
                                        { s "nature physics" =
                                            { "Nat. Phys." }
                                            { s "proceedings of the national academy of sciences of the united states of america" =
                                                { "Proc. Natl. Acad. Sci. USA" }
                                                { s "review of scientific instruments" =
                                                    { "Rev. Sci. Instrum." }
                                                    { s "journal of vacuum science & technology" =
                                                        { "J. Vac. Sci. Technol." }
                                                        { s "nature" =
                                                            { "Nature" }
                                                            { s "science" =
                                                                { "Science" }
                                                                { journal } % already abbreviated or unknown
                                                              if$
                                                            }
                                                          if$
                                                        }
                                                      if$
                                                    }
                                                  if$
                                                }
                                              if$
                                            }
                                          if$
                                        }
                                      if$
                                    }
                                  if$
                                }
                              if$
                            }
                          if$
                        }
                      if$
                    }
                  if$
                }
              if$
            }
          if$
        }
      if$
    }
  if$
}

% ======= TYPES =======

FUNCTION {article}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check new.comma
  crossref missing$
    { journal.to.abbrev
      emphasize
      "journal" output.check
      add.blank
      format.vol.num.pages output
      format.date "year" output.check
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  new.sentence
  fin.entry
}

FUNCTION {book}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$

  format.btitle "title" output.check         % Title of Book（italic, no quotes）
  format.edition.short output                % , 2nd ed.
  format.book.paren duplicate$ empty$
   'pop$
   { " " swap$ * before.all 'output.state := output.nonnull }
  if$

  new.sentence
  fin.entry
}




FUNCTION {booklet}
{ output.bibitem
  format.authors output
  format.title "title" output.check
  howpublished output
  address output
  format.date output
  new.sentence
  fin.entry
}

FUNCTION {inbook}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  format.btitle "title" output.check
  crossref missing$
    {
      format.number.series output
      format.publisher.address output   % intentionally empty
      format.bvolume output
      format.chapter.pages "chapter and pages" output.check
    }
    {
      format.chapter.pages "chapter and pages" output.check
      format.book.crossref output.nonnull
      format.date "year" output.check
    }
  if$
  format.edition output
  new.sentence
  
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.number.series output
      format.publisher.address output   % intentionally empty
      format.bvolume output
      format.chapter.pages output
      format.edition output
    }
    { format.incoll.inproc.crossref output.nonnull
      format.chapter.pages output
    }
  if$
  new.sentence
  
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check

  % 追加：著者の直後に "Title", Conference name (booktitle)
  format.title "title" output.check
  crossref missing$
    { format.in.ed.proctitle "booktitle" output.check

      format.number.series output
    % publisher/organization は出さない（要望どおり）
    format.bvolume output
    format.pages output
    "" format.org.or.pub output   % ← 常に (address, year) を試みる

    }
    { format.incoll.inproc.crossref output.nonnull
      format.pages output
    }
  if$
  new.sentence
  fin.entry
}


FUNCTION {conference} { inproceedings }

FUNCTION {manual}
{ output.bibitem
  author empty$
    { organization empty$
        'skip$
        { organization output.nonnull
          address output
        }
      if$
    }
    { format.authors output.nonnull }
  if$
  format.btitle "title" output.check
  author empty$
    { organization empty$
    {
          address output
        }
        'skip$
      if$
    }
    {
      organization output
      address output
    }
  if$
  format.edition output
  format.date output
  new.sentence
  
  fin.entry
}

FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  bbl.mthesis format.thesis.type output.nonnull
  school "school" output.check
  address output
  format.date "year" output.check
  new.sentence
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.authors output
  is.arxiv
    { format.title "title" output.check     % ← "Title" の形式
      new.comma
      format.arxiv.id output                % ← arXiv:....
      fin.entry                             % ←年は出さない
    }
    { format.title.misc output              % ←通常misc（Web等）
      format.url output
      howpublished output
      format.date output
      new.sentence
      fin.entry
      empty.misc.check
    }
  if$
}


FUNCTION {phdthesis}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  bbl.phdthesis format.thesis.type output.nonnull
  school "school" output.check
  address output
  format.date "year" output.check
  new.sentence
  fin.entry
}

FUNCTION {proceedings}
{ output.bibitem
  editor empty$
    { organization output }
    { format.editors output.nonnull }
  if$
  format.btitle "title" output.check
  format.bvolume output
  editor empty$
    { publisher empty$
        'skip$
        {
          format.number.series output
          format.publisher.address output   % intentionally empty
        }
      if$
    }
    { publisher empty$
        { format.organization.address output }
        {
          organization output
          format.publisher.address output   % intentionally empty
        }
      if$
     }
  if$
  new.sentence
  
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  format.tr.number emphasize output.nonnull
  institution "institution" output.check
  address output
  format.date "year" output.check
  new.sentence
  fin.entry
}

FUNCTION {unpublished}
{ output.bibitem
  format.authors "author" output.check
  format.title "title" output.check
  format.date output
  new.sentence
  format.note "note" output.check
  fin.entry
}

FUNCTION {default.type} { misc }

READ

STRINGS { longest.label }

INTEGERS { number.label longest.label.width }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #1 'number.label :=
  #0 'longest.label.width :=
}

FUNCTION {longest.label.pass}
{ number.label int.to.str$ 'label :=
  number.label #1 + 'number.label :=
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
}

EXECUTE {initialize.longest.label}

ITERATE {longest.label.pass}

FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{"  longest.label  * "}" *
  write$ newline$
}

EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}
%% End of file `ScienceAdvances_custom.bst'.
