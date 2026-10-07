/*  g2ntotex.c   
 *  Usage:  g2ntotex fontname (converts fontname.g2n to fontname.tex)
 *
 *  Copyright 2012 Bob Tennent
 *  School of Computing, Queen's University
 *  This program is free software, licensed under the GNU GPL, >=2.0.
 *  This software comes with absolutely NO WARRANTY.
 *                                         */
# include <stdlib.h>  
# include <stdio.h>
# include <string.h>
# define FILE_LEN 36
# define LINE_LEN 72

FILE *g2n_file, *tex_file;
char g2n_filename[FILE_LEN+5];
char tex_filename[FILE_LEN+5];

void error (char msg[])  /* abort with stderr message msg */
{ fprintf (stderr, "Error: %s\n", msg);
  exit (EXIT_FAILURE);
}

int main (int argc, char *argv[]) {
  int ch;
  int gid;
  char codepoint[5];
  char glyphname [LINE_LEN+1];
  g2n_filename[0] = '\0';
  strncat (g2n_filename, argv[1], FILE_LEN);
  strcat (g2n_filename, ".g2n");
  g2n_file = fopen (g2n_filename, "r+");
  if (g2n_file == NULL) error("Can't open g2n file.");
  tex_filename[0] = '\0';
  strncat (tex_filename, argv[1], FILE_LEN);
  strcat (tex_filename, ".tex");
  tex_file = fopen (tex_filename, "w+");
  if (tex_file == NULL) error("Can't open tex file.");
  /* skip first line: */
  ch = getc (g2n_file); while (ch != '\n') { ch = getc (g2n_file);}  
  while (fscanf (g2n_file, 
                 "GLYPHID%i PSNAME %s UNICODE %s\n", 
                 &gid, &glyphname, &codepoint)
          == 3 ) 
  { int glyphindex;
    sscanf ( codepoint, "%x", &glyphindex);   /*  convert hex codepoint to integer index */
    if (glyphname[0] != 'u' || glyphname[1] != 'n' || glyphname[2] != 'i' ) {
      fprintf (tex_file, "\\DeclareTextGlyphY{%s}{%s}{%i}\n", argv[1], glyphname, glyphindex);
    }
    fprintf (tex_file, "\\DeclareTextGlyphY{%s}{uni%s}{%i}\n", argv[1], codepoint, glyphindex);
  }
  return 0;
}
